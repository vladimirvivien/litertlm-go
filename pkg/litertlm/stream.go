package litertlm

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/vladimirvivien/litertlm-go/pkg/utils"
)

// StreamChunk is one piece of a streaming generation result. A callback will
// receive multiple non-Final chunks followed by a single Final chunk. If the
// underlying C layer reports an error, Err is populated and Final is true.
type StreamChunk struct {
	Text  string
	Final bool
	Err   error
}

// ---- One permanent trampoline shared by all streams ----------------------
//
// purego.NewCallback allocates a C function pointer that lives for the rest
// of the process. Registering a new callback per inference call would leak
// memory, so we register exactly one trampoline function and dispatch by a
// cookie passed through the C `callback_data` argument.

var (
	streamOnce     sync.Once
	streamTrampAddr uintptr

	streamRegMu sync.Mutex
	streamRegM  = map[uintptr]func(StreamChunk){}
	streamRegID uintptr
)

// streamTrampoline runs on whatever thread C invokes the callback from;
// purego handles the cgo-like transition into the Go runtime for us.
//
// The pointer arguments are declared as unsafe.Pointer (rather than uintptr)
// so converting them to *byte does not trip `go vet`'s
// "possible misuse of unsafe.Pointer" heuristic.
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

// ---- Public streaming entry points ---------------------------------------

// GenerateContentStream starts a streaming generation. The callback is
// invoked on a background thread for each chunk; the final invocation has
// Final=true (with Err set if the C layer reported an error).
//
// The call blocks until the Final chunk is delivered to cb, so callers can
// treat the function synchronously. Run it in a goroutine to consume the
// stream concurrently with other work.
func (s Session) GenerateContentStream(inputs []InputData, cb func(StreamChunk)) error {
	if s == 0 {
		return errors.New("litertlm: generate_content_stream: invalid session")
	}
	if len(inputs) == 0 {
		return errors.New("litertlm: generate_content_stream: no inputs")
	}
	if cb == nil {
		return errors.New("litertlm: generate_content_stream: nil callback")
	}

	done := make(chan struct{})
	id := registerStreamCB(func(sc StreamChunk) {
		cb(sc)
		if sc.Final {
			close(done)
		}
	})

	inputsPtr := unsafe.Pointer(&inputs[0])
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
	runtime.KeepAlive(inputs)

	if ret != 0 {
		unregisterStreamCB(id)
		return fmt.Errorf("litertlm: generate_content_stream start failed (code=%d)", ret)
	}

	<-done
	unregisterStreamCB(id)
	return nil
}

// GenerateContentStreamCh is the channel-idiomatic convenience wrapper over
// GenerateContentStream. The returned channel is closed after the Final
// chunk has been sent.
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

// ConversationSendMessageStream is the Conversation-level streaming send.
// Mirrors the signature of Session.GenerateContentStream.
func (c Conversation) SendMessageStream(messageJSON, extraContext string, cb func(StreamChunk)) error {
	if c == 0 {
		return errors.New("litertlm: send_message_stream: invalid conversation")
	}
	if cb == nil {
		return errors.New("litertlm: send_message_stream: nil callback")
	}

	msgPtr, err := utils.BytePtrFromString(messageJSON)
	if err != nil {
		return err
	}
	var ctxPtr *byte
	if extraContext != "" {
		ctxPtr, err = utils.BytePtrFromString(extraContext)
		if err != nil {
			return err
		}
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
		unsafe.Pointer(&cbAddr),
		unsafe.Pointer(&cbData),
	)

	if ret != 0 {
		unregisterStreamCB(id)
		return fmt.Errorf("litertlm: send_message_stream start failed (code=%d)", ret)
	}
	<-done
	unregisterStreamCB(id)
	return nil
}

// SendMessageStreamCh is the channel variant of SendMessageStream.
func (c Conversation) SendMessageStreamCh(messageJSON, extraContext string) <-chan StreamChunk {
	out := make(chan StreamChunk, 16)
	go func() {
		defer close(out)
		err := c.SendMessageStream(messageJSON, extraContext, func(sc StreamChunk) {
			out <- sc
		})
		if err != nil {
			out <- StreamChunk{Final: true, Err: err}
		}
	}()
	return out
}
