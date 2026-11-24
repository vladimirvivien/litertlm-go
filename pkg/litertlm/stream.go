package litertlm

import (
	"fmt"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/vladimirvivien/litertlm-go/internal/cstr"
)

// streamWatchdog bounds the wait for the C engine's Final chunk so the
// Go runtime stays runnable while a consumer waits on a foreign C++ thread.
const streamWatchdog = 24 * time.Hour

// StreamChunk is one piece of a streaming generation result. Callbacks
// receive zero or more non-Final chunks followed by a Final chunk;
// on error, Err is populated and Final is true.
type StreamChunk struct {
	Text  string
	Final bool
	Err   error
}

// callbackRegistry maps uintptr cookies passed across FFI user_data pointers
// back to strongly-typed Go callbacks.
type callbackRegistry[T any] struct {
	mu    sync.RWMutex
	next  uintptr
	items map[uintptr]T
}

func (r *callbackRegistry[T]) Register(cb T) uintptr {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.items == nil {
		r.items = make(map[uintptr]T)
	}
	r.next++
	id := r.next
	r.items[id] = cb
	return id
}

func (r *callbackRegistry[T]) Lookup(id uintptr) (T, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	cb, ok := r.items[id]
	return cb, ok
}

func (r *callbackRegistry[T]) Unregister(id uintptr) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.items, id)
}

var (
	streamOnce      sync.Once
	streamTrampAddr uintptr
	streamRegistry  callbackRegistry[func(StreamChunk)]
)

// streamTrampoline matches LiteRtLmStreamCallback:
// void (*)(void* callback_data, const LiteRtLmStreamChunk* chunk)
func streamTrampoline(data uintptr, chunk uintptr) uintptr {
	cb, ok := streamRegistry.Lookup(data)
	if !ok || cb == nil || chunk == 0 {
		return 0
	}

	var finalSt, textSt, errSt int32
	var isFinal uint8
	isFinalPtr := &isFinal
	finalCallErr := streamChunkIsFinalFunc.callErr(unsafe.Pointer(&finalSt), unsafe.Pointer(&chunk), unsafe.Pointer(&isFinalPtr))
	runtime.KeepAlive(isFinalPtr)

	var textPtr *byte
	textOutPtr := &textPtr
	textCallErr := streamChunkGetTextFunc.callErr(unsafe.Pointer(&textSt), unsafe.Pointer(&chunk), unsafe.Pointer(&textOutPtr))
	runtime.KeepAlive(textOutPtr)

	var errPtr *byte
	errOutPtr := &errPtr
	errCallErr := streamChunkGetErrorFunc.callErr(unsafe.Pointer(&errSt), unsafe.Pointer(&chunk), unsafe.Pointer(&errOutPtr))
	runtime.KeepAlive(errOutPtr)

	sc := StreamChunk{Final: finalCallErr == nil && finalSt == 0 && isFinal != 0}
	if textCallErr == nil && textSt == 0 && textPtr != nil {
		sc.Text = cstr.BytePtrToString(textPtr)
	}
	switch {
	case errCallErr != nil:
		sc.Err = errCallErr
		sc.Final = true
	case finalCallErr != nil:
		sc.Err = finalCallErr
		sc.Final = true
	case textCallErr != nil:
		sc.Err = textCallErr
		sc.Final = true
	case errSt == 0 && errPtr != nil:
		sc.Err = &CError{
			Op:      "stream_chunk",
			Code:    StatusUnknown,
			Message: cstr.BytePtrToString(errPtr),
		}
		sc.Final = true
	}
	cb(sc)
	return 0
}

func ensureStreamTrampoline() {
	streamOnce.Do(func() {
		streamTrampAddr = purego.NewCallback(streamTrampoline)
	})
}

func runStream(op string, cb func(StreamChunk), start func(cbAddr, cbData uintptr) error) error {
	ensureStreamTrampoline()
	done := make(chan struct{})
	var doneOnce sync.Once
	id := streamRegistry.Register(func(sc StreamChunk) {
		cb(sc)
		if sc.Final {
			doneOnce.Do(func() { close(done) })
		}
	})
	defer streamRegistry.Unregister(id)

	if err := start(streamTrampAddr, id); err != nil {
		return err
	}

	watchdog := time.NewTimer(streamWatchdog)
	defer watchdog.Stop()

	select {
	case <-done:
		return nil
	case <-watchdog.C:
		return fmt.Errorf("litertlm: %s: watchdog fired after %v", op, streamWatchdog)
	}
}

func streamToChan(run func(cb func(StreamChunk)) error) <-chan StreamChunk {
	out := make(chan StreamChunk, 16)
	go func() {
		defer close(out)
		if err := run(func(sc StreamChunk) {
			out <- sc
		}); err != nil {
			out <- StreamChunk{Final: true, Err: err}
		}
	}()
	return out
}

// GenerateContentStream starts a streaming generation. The callback is
// invoked on a background thread for each chunk; the final invocation
// has Final=true. The call blocks until the Final chunk has been
// delivered to cb.
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

	ptrs := handlePtrs(inputs)
	inputsPtr := unsafe.Pointer(&ptrs[0])
	n := uint64(len(inputs))

	err := runStream("generate_content_stream", cb, func(cbAddr, cbData uintptr) error {
		return callStatus(sessionGenerateContentStreamFunc, "session_generate_content_stream",
			unsafe.Pointer(&s),
			unsafe.Pointer(&inputsPtr),
			unsafe.Pointer(&n),
			unsafe.Pointer(&cbAddr),
			unsafe.Pointer(&cbData),
		)
	})
	runtime.KeepAlive(ptrs)
	return err
}

// GenerateContentStreamCh is the channel-idiomatic wrapper over
// GenerateContentStream. The channel is closed after Final is sent.
func (s Session) GenerateContentStreamCh(inputs []InputData) <-chan StreamChunk {
	return streamToChan(func(cb func(StreamChunk)) error {
		return s.GenerateContentStream(inputs, cb)
	})
}

// RunDecodeAsync starts non-blocking decoding for inputs previously added via
// Session.RunPrefill, delivering chunks to cb until Final=true.
func (s Session) RunDecodeAsync(cb func(StreamChunk)) error {
	if s == 0 {
		return fmt.Errorf("litertlm: run_decode_async: invalid session")
	}
	if cb == nil {
		return fmt.Errorf("litertlm: run_decode_async: nil callback")
	}
	return runStream("run_decode_async", cb, func(cbAddr, cbData uintptr) error {
		return callStatus(sessionRunDecodeAsyncFunc, "session_run_decode_async",
			unsafe.Pointer(&s),
			unsafe.Pointer(&cbAddr),
			unsafe.Pointer(&cbData),
		)
	})
}

// RunDecodeAsyncCh is the channel-idiomatic wrapper over RunDecodeAsync.
// The returned channel closes after the Final chunk is delivered.
func (s Session) RunDecodeAsyncCh() <-chan StreamChunk {
	return streamToChan(func(cb func(StreamChunk)) error {
		return s.RunDecodeAsync(cb)
	})
}

// SendMessageStream is the Conversation-level streaming send. Mirrors
// Session.GenerateContentStream. opts carries per-call knobs such as
// the visual token budget; pass OptionalArgs(0) for C-side defaults.
func (c Conversation) SendMessageStream(messageJSON, extraContext string, opts OptionalArgs, cb func(StreamChunk)) error {
	return c.sendMessageStream(messageJSON, extraContext, opts, false, cb)
}

func (c Conversation) sendMessageStream(messageJSON, extraContext string, opts OptionalArgs, driveWait bool, cb func(StreamChunk)) error {
	if c == 0 {
		return fmt.Errorf("litertlm: send_message_stream: invalid conversation")
	}
	if cb == nil {
		return fmt.Errorf("litertlm: send_message_stream: nil callback")
	}

	msgPtr, err := cstr.BytePtrFromString(messageJSON)
	if err != nil {
		return err
	}
	ctxPtr, err := bytePtrOrNil(extraContext)
	if err != nil {
		return err
	}

	err = runStream("send_message_stream", cb, func(cbAddr, cbData uintptr) error {
		if callErr := callStatus(conversationSendMessageStreamFunc, "conversation_send_message_stream",
			unsafe.Pointer(&c),
			unsafe.Pointer(&msgPtr),
			unsafe.Pointer(&ctxPtr),
			unsafe.Pointer(&opts),
			unsafe.Pointer(&cbAddr),
			unsafe.Pointer(&cbData),
		); callErr != nil {
			return callErr
		}
		if driveWait {
			return c.WaitUntilDone()
		}
		return nil
	})
	// Hold the C strings live until Final has been delivered.
	runtime.KeepAlive(msgPtr)
	runtime.KeepAlive(ctxPtr)
	return err
}

// SendMessageStreamCh is the channel variant of SendMessageStream.
// opts carries per-call knobs; pass OptionalArgs(0) for C-side defaults.
func (c Conversation) SendMessageStreamCh(messageJSON, extraContext string, opts OptionalArgs) <-chan StreamChunk {
	return c.sendMessageStreamCh(messageJSON, extraContext, opts, false)
}

func (c Conversation) sendMessageStreamCh(messageJSON, extraContext string, opts OptionalArgs, driveWait bool) <-chan StreamChunk {
	return streamToChan(func(cb func(StreamChunk)) error {
		return c.sendMessageStream(messageJSON, extraContext, opts, driveWait, cb)
	})
}
