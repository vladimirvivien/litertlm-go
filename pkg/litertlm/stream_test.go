package litertlm

import (
	"errors"
	"strings"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/jupiterrider/ffi"
)

func TestStreamTrampoline_OpaqueChunk(t *testing.T) {
	isFinalCb := purego.NewCallback(func(chunk uintptr, out *uint8) uintptr {
		if chunk == 0x10 {
			*out = 0
		} else {
			*out = 1
		}
		return uintptr(StatusOk)
	})
	textBytes := append([]byte("hello "), 0)
	textCb := purego.NewCallback(func(chunk uintptr, out **byte) uintptr {
		if chunk == 0x10 {
			*out = &textBytes[0]
		} else {
			*out = nil
		}
		return uintptr(StatusOk)
	})
	errBytes := append([]byte("stream aborted"), 0)
	errCb := purego.NewCallback(func(chunk uintptr, out **byte) uintptr {
		if chunk == 0x20 {
			*out = &errBytes[0]
		} else {
			*out = nil
		}
		return uintptr(StatusOk)
	})

	var cif ffi.Cif
	if st := ffi.PrepCif(&cif, ffi.DefaultAbi, 2, &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer); st != ffi.OK {
		t.Fatalf("PrepCif: %v", st)
	}

	origFinal := streamChunkIsFinalFunc
	origText := streamChunkGetTextFunc
	origErr := streamChunkGetErrorFunc
	t.Cleanup(func() {
		streamChunkIsFinalFunc = origFinal
		streamChunkGetTextFunc = origText
		streamChunkGetErrorFunc = origErr
	})

	makeBound := func(name string, addr uintptr) *lazyFun {
		lf := &lazyFun{
			name: name,
			ret:  &ffi.TypeSint32,
			args: []*ffi.Type{&ffi.TypePointer, &ffi.TypePointer},
			fun:  ffi.Fun{Addr: addr, Cif: &cif},
		}
		lf.once.Do(func() {})
		return lf
	}
	streamChunkIsFinalFunc = makeBound("litert_lm_stream_chunk_is_final", isFinalCb)
	streamChunkGetTextFunc = makeBound("litert_lm_stream_chunk_get_text", textCb)
	streamChunkGetErrorFunc = makeBound("litert_lm_stream_chunk_get_error", errCb)

	var got []StreamChunk
	id := streamRegistry.Register(func(sc StreamChunk) {
		got = append(got, sc)
	})
	defer streamRegistry.Unregister(id)

	streamTrampoline(id, 0x10)
	streamTrampoline(id, 0x20)

	if len(got) != 2 {
		t.Fatalf("received %d chunks, want 2", len(got))
	}
	if got[0].Text != "hello " || got[0].Final || got[0].Err != nil {
		t.Errorf("chunk 0 = %+v, want {Text:\"hello \" Final:false Err:nil}", got[0])
	}
	if got[1].Text != "" || !got[1].Final || !errors.Is(got[1].Err, ErrUnknown) {
		t.Errorf("chunk 1 = %+v, want final ErrUnknown", got[1])
	}
	var cerr *CError
	if !errors.As(got[1].Err, &cerr) || !strings.Contains(cerr.Error(), "stream aborted") {
		t.Errorf("chunk 1 err = %v, want *CError mentioning stream aborted", got[1].Err)
	}
	_ = unsafe.Pointer(&textBytes[0])
	_ = unsafe.Pointer(&errBytes[0])
}

func TestCallbackRegistry_RegisterLookupUnregister(t *testing.T) {
	var reg callbackRegistry[string]
	id1 := reg.Register("first")
	id2 := reg.Register("second")
	if id1 == id2 {
		t.Fatalf("expected distinct callback IDs, got %d", id1)
	}
	if v, ok := reg.Lookup(id1); !ok || v != "first" {
		t.Errorf("Lookup(%d) = (%q, %v), want (first, true)", id1, v, ok)
	}
	reg.Unregister(id1)
	if _, ok := reg.Lookup(id1); ok {
		t.Errorf("Lookup(%d) succeeded after Unregister", id1)
	}
	if v, ok := reg.Lookup(id2); !ok || v != "second" {
		t.Errorf("Lookup(%d) = (%q, %v), want (second, true)", id2, v, ok)
	}
}

func TestRunStream_DoubleFinalDoesNotPanic(t *testing.T) {
	var received []StreamChunk
	err := runStream("test_double_final", func(sc StreamChunk) {
		received = append(received, sc)
	}, func(_, cbData uintptr) error {
		cb, ok := streamRegistry.Lookup(cbData)
		if !ok {
			t.Fatal("callback missing")
		}
		cb(StreamChunk{Text: "end-1", Final: true})
		cb(StreamChunk{Text: "end-2", Final: true})
		return nil
	})
	if err != nil {
		t.Fatalf("runStream failed: %v", err)
	}
	if len(received) != 2 {
		t.Errorf("got %d chunks, want 2", len(received))
	}
}
