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

func TestLazyFun_MissingSymbolReturnsCError(t *testing.T) {
	wantErr := errors.New("symbol not found")
	withStubPrep(t, func(name string, ret *ffi.Type, args ...*ffi.Type) (ffi.Fun, error) {
		return ffi.Fun{}, wantErr
	})

	f := newLazyFun("litert_lm_definitely_missing", &ffi.TypeVoid)

	err := f.callErr(nil)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !errors.Is(err, ErrUnimplemented) {
		t.Errorf("errors.Is(err, ErrUnimplemented) = false; got %v", err)
	}
	if !errors.Is(err, wantErr) {
		t.Errorf("errors.Is(err, wantErr) = false; got %v", err)
	}
	msg := err.Error()
	if !strings.Contains(msg, "litert_lm_definitely_missing") {
		t.Errorf("error message must name the missing symbol, got: %q", msg)
	}
	if !strings.Contains(msg, "litertlm:") {
		t.Errorf("error message should be prefixed `litertlm:`, got: %q", msg)
	}
}

func TestLazyFun_StickyAndPrepCalledOnce(t *testing.T) {
	var prepCalls atomic.Int32
	withStubPrep(t, func(name string, ret *ffi.Type, args ...*ffi.Type) (ffi.Fun, error) {
		prepCalls.Add(1)
		return ffi.Fun{}, errors.New("symbol not found")
	})

	f := newLazyFun("litert_lm_definitely_missing_2", &ffi.TypeVoid)

	if err := f.callErr(nil); !errors.Is(err, ErrUnimplemented) {
		t.Fatalf("first call: got %v, want ErrUnimplemented", err)
	}
	if err := f.callErr(nil); !errors.Is(err, ErrUnimplemented) {
		t.Fatalf("second call: got %v, want sticky ErrUnimplemented", err)
	}
	if got := prepCalls.Load(); got != 1 {
		t.Errorf("prepSymbol should be invoked exactly once across two calls (sync.Once), got %d", got)
	}
}
