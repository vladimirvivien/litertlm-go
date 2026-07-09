package litertlm

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/vladimirvivien/litertlm-go/pkg/utils"
)

// streamWatchdog bounds the wait for the C engine's Final chunk.
// Its real job is to keep a runnable timer on the scheduler so the
// Go runtime does not fire an "all goroutines asleep" deadlock
// while the consumer is parked on <-done waiting for a foreign-thread
// callback.
const streamWatchdog = 24 * time.Hour

// StreamChunk is one piece of a streaming generation result. Callbacks
// receive zero or more non-Final chunks followed by exactly one Final
// chunk; on error, Err is populated and Final is true.
type StreamChunk struct {
	Text  string
	Final bool
	Err   error
}

// One process-wide trampoline shared by all streams. purego.NewCallback
// allocates a permanent C function pointer, so a single trampoline is
// registered and streams dispatch by a cookie passed through
// `callback_data`.
var (
	streamOnce      sync.Once
	streamTrampAddr uintptr

	streamRegMu sync.Mutex
	streamRegM  = map[uintptr]func(StreamChunk){}
	streamRegID uintptr
)

func streamTrampoline(data uintptr, chunk unsafe.Pointer, isFinal uint8, errMsg unsafe.Pointer) uintptr {
	streamRegMu.Lock()
	cb := streamRegM[data]
	streamRegMu.Unlock()
	if cb == nil {
		return 0
	}

	sc := StreamChunk{Final: isFinal != 0}
	if chunk != nil {
		sc.Text = utils.BytePtrToString((*byte)(chunk))
	}
	if errMsg != nil {
		sc.Err = errors.New(utils.BytePtrToString((*byte)(errMsg)))
	}
	cb(sc)
	return 0
}

func ensureStreamTrampoline() {
	streamOnce.Do(func() {
		streamTrampAddr = purego.NewCallback(streamTrampoline)
	})
}

func registerStreamCB(cb func(StreamChunk)) uintptr {
	ensureStreamTrampoline()
	streamRegMu.Lock()
	streamRegID++
	id := streamRegID
	streamRegM[id] = cb
	streamRegMu.Unlock()
	return id
}

func unregisterStreamCB(id uintptr) {
	streamRegMu.Lock()
	delete(streamRegM, id)
	streamRegMu.Unlock()
}

// GenerateContentStream starts a streaming generation. The callback is
// invoked on a background thread for each chunk; the final invocation
// has Final=true. The call blocks until the Final chunk has been
// delivered to cb, so callers can treat it synchronously.
func (s Session) GenerateContentStream(inputs []InputData, cb func(StreamChunk)) error {
	if s == 0 {
		return fmt.Errorf("litertlm: generate_content_stream: invalid session")
	}
	if len(inputs) == 0 {
		return fmt.Errorf("litertlm: generate_content_stream: no inputs")
	}
	if cb == nil {
		return fmt.Errorf("litertlm: generate_content_stream: nil callback")
	}

	done := make(chan struct{})
	id := registerStreamCB(func(sc StreamChunk) {
		cb(sc)
		if sc.Final {
			close(done)
		}
	})

	ptrs := make([]uintptr, len(inputs))
	for i, in := range inputs {
		ptrs[i] = uintptr(in)
	}
	inputsPtr := unsafe.Pointer(&ptrs[0])
	n := uint64(len(inputs))
	cbAddr := streamTrampAddr
	cbData := id

	var ret int32
	sessionGenerateContentStreamFunc.Call(
		unsafe.Pointer(&ret),
		unsafe.Pointer(&s),
		unsafe.Pointer(&inputsPtr),
		unsafe.Pointer(&n),
		unsafe.Pointer(&cbAddr),
		unsafe.Pointer(&cbData),
	)

	if ret != 0 {
		unregisterStreamCB(id)
		runtime.KeepAlive(ptrs)
		return fmt.Errorf("litertlm: generate_content_stream start failed (code=%d)", ret)
	}

	watchdog := time.NewTimer(streamWatchdog)
	defer watchdog.Stop()

	select {
	case <-done:
	case <-watchdog.C:
		unregisterStreamCB(id)
		runtime.KeepAlive(ptrs)
		return fmt.Errorf("litertlm: generate_content_stream: watchdog fired after %v", streamWatchdog)
	}
	unregisterStreamCB(id)
	runtime.KeepAlive(ptrs)
	return nil
}

// GenerateContentStreamCh is the channel-idiomatic wrapper over
// GenerateContentStream. The channel is closed after Final is sent.
//
//	for chunk := range session.GenerateContentStreamCh(inputs) {
//	    if chunk.Err != nil { ... }
//	    fmt.Print(chunk.Text)
//	}
func (s Session) GenerateContentStreamCh(inputs []InputData) <-chan StreamChunk {
	out := make(chan StreamChunk, 16)
	go func() {
		defer close(out)
		err := s.GenerateContentStream(inputs, func(sc StreamChunk) {
			out <- sc
		})
		if err != nil {
			out <- StreamChunk{Final: true, Err: err}
		}
	}()
	return out
}

// SendMessageStream is the Conversation-level streaming send. Mirrors
// Session.GenerateContentStream. opts carries per-call knobs such as
// the visual token budget; pass OptionalArgs(0) for C-side defaults.
func (c Conversation) SendMessageStream(messageJSON, extraContext string, opts OptionalArgs, cb func(StreamChunk)) error {
	if c == 0 {
		return fmt.Errorf("litertlm: send_message_stream: invalid conversation")
	}
	if cb == nil {
		return fmt.Errorf("litertlm: send_message_stream: nil callback")
	}

	msgPtr, err := utils.BytePtrFromString(messageJSON)
	if err != nil {
		return err
	}
	ctxPtr, err := bytePtrOrNil(extraContext)
	if err != nil {
		return err
	}

	done := make(chan struct{})
	id := registerStreamCB(func(sc StreamChunk) {
		cb(sc)
		if sc.Final {
			close(done)
		}
	})

	cbAddr := streamTrampAddr
	cbData := id

	var ret int32
	conversationSendMessageStreamFunc.Call(
		unsafe.Pointer(&ret),
		unsafe.Pointer(&c),
		unsafe.Pointer(&msgPtr),
		unsafe.Pointer(&ctxPtr),
		unsafe.Pointer(&opts),
		unsafe.Pointer(&cbAddr),
		unsafe.Pointer(&cbData),
	)

	if ret != 0 {
		unregisterStreamCB(id)
		runtime.KeepAlive(msgPtr)
		runtime.KeepAlive(ctxPtr)
		return fmt.Errorf("litertlm: send_message_stream start failed (code=%d)", ret)
	}

	watchdog := time.NewTimer(streamWatchdog)
	defer watchdog.Stop()

	select {
	case <-done:
	case <-watchdog.C:
		unregisterStreamCB(id)
		runtime.KeepAlive(msgPtr)
		runtime.KeepAlive(ctxPtr)
		return fmt.Errorf("litertlm: send_message_stream: watchdog fired after %v", streamWatchdog)
	}
	unregisterStreamCB(id)
	// Hold the C strings live until Final has been delivered.
	runtime.KeepAlive(msgPtr)
	runtime.KeepAlive(ctxPtr)
	return nil
}

// SendMessageStreamCh is the channel variant of SendMessageStream.
// opts carries per-call knobs; pass OptionalArgs(0) for C-side defaults.
func (c Conversation) SendMessageStreamCh(messageJSON, extraContext string, opts OptionalArgs) <-chan StreamChunk {
	out := make(chan StreamChunk, 16)
	go func() {
		defer close(out)
		err := c.SendMessageStream(messageJSON, extraContext, opts, func(sc StreamChunk) {
			out <- sc
		})
		if err != nil {
			out <- StreamChunk{Final: true, Err: err}
		}
	}()
	return out
}
