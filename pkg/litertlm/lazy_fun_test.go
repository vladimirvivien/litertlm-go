package litertlm

import (
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/jupiterrider/ffi"
)

// withStubPrep swaps prepSymbol for the duration of the test, restoring
// it via t.Cleanup so subsequent tests see the real resolver.
func withStubPrep(t *testing.T, stub func(name string, ret *ffi.Type, args ...*ffi.Type) (ffi.Fun, error)) {
	t.Helper()
	prev := prepSymbol
	prepSymbol = stub
	t.Cleanup(func() { prepSymbol = prev })
}

func TestLazyFun_PanicsWithSymbolName(t *testing.T) {
	wantErr := errors.New("symbol not found")
	withStubPrep(t, func(name string, ret *ffi.Type, args ...*ffi.Type) (ffi.Fun, error) {
		return ffi.Fun{}, wantErr
	})

	f := newLazyFun("litert_lm_definitely_missing", &ffi.TypeVoid)

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic, got none")
		}
		err, ok := r.(error)
		if !ok {
			t.Fatalf("panic should carry an error, got %T (%v)", r, r)
		}
		msg := err.Error()
		if !strings.Contains(msg, "litert_lm_definitely_missing") {
			t.Errorf("panic message must name the missing symbol, got: %q", msg)
		}
		if !strings.Contains(msg, "litertlm:") {
			t.Errorf("panic message should be prefixed `litertlm:`, got: %q", msg)
		}
		if !errors.Is(err, wantErr) {
			t.Errorf("panic error should wrap underlying Prep error, got: %q", msg)
		}
	}()
	f.Call(nil)
}

func TestLazyFun_StickyAndPrepCalledOnce(t *testing.T) {
	var prepCalls atomic.Int32
	withStubPrep(t, func(name string, ret *ffi.Type, args ...*ffi.Type) (ffi.Fun, error) {
		prepCalls.Add(1)
		return ffi.Fun{}, errors.New("symbol not found")
	})

	f := newLazyFun("litert_lm_definitely_missing_2", &ffi.TypeVoid)

	callExpectingPanic := func() (panicked bool) {
		defer func() {
			if r := recover(); r != nil {
				panicked = true
			}
		}()
		f.Call(nil)
		return false
	}

	if !callExpectingPanic() {
		t.Fatal("first call: expected panic")
	}
	if !callExpectingPanic() {
		t.Fatal("second call: expected panic (recorded error must be sticky)")
	}
	if got := prepCalls.Load(); got != 1 {
		t.Errorf("prepSymbol should be invoked exactly once across two Calls (sync.Once), got %d", got)
	}
}
