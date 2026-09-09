package litertlm

import (
	"errors"
	"strings"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/jupiterrider/ffi"
)

func TestStatusCode_ValuesAndStrings(t *testing.T) {
	tests := []struct {
		code     StatusCode
		wantInt  int32
		wantName string
		wantErr  error
	}{
		{StatusOk, 0, "OK", nil},
		{StatusCancelled, 1, "Cancelled", ErrCancelled},
		{StatusUnknown, 2, "Unknown", ErrUnknown},
		{StatusInvalidArgument, 3, "InvalidArgument", ErrInvalidArgument},
		{StatusDeadlineExceeded, 4, "DeadlineExceeded", ErrDeadlineExceeded},
		{StatusNotFound, 5, "NotFound", ErrNotFound},
		{StatusAlreadyExists, 6, "AlreadyExists", ErrAlreadyExists},
		{StatusPermissionDenied, 7, "PermissionDenied", ErrPermissionDenied},
		{StatusResourceExhausted, 8, "ResourceExhausted", ErrResourceExhausted},
		{StatusFailedPrecondition, 9, "FailedPrecondition", ErrFailedPrecondition},
		{StatusAborted, 10, "Aborted", ErrAborted},
		{StatusOutOfRange, 11, "OutOfRange", ErrOutOfRange},
		{StatusUnimplemented, 12, "Unimplemented", ErrUnimplemented},
		{StatusInternal, 13, "Internal", ErrInternal},
		{StatusUnavailable, 14, "Unavailable", ErrUnavailable},
		{StatusDataLoss, 15, "DataLoss", ErrDataLoss},
		{StatusUnauthenticated, 16, "Unauthenticated", ErrUnauthenticated},
	}

	for _, tc := range tests {
		if int32(tc.code) != tc.wantInt {
			t.Errorf("%s = %d, want %d", tc.wantName, tc.code, tc.wantInt)
		}
		if got := tc.code.String(); got != tc.wantName {
			t.Errorf("StatusCode(%d).String() = %q, want %q", tc.wantInt, got, tc.wantName)
		}
		if !errors.Is(tc.code.Err(), tc.wantErr) {
			t.Errorf("StatusCode(%d).Err() = %v, want %v", tc.wantInt, tc.code.Err(), tc.wantErr)
		}
	}

	if got := StatusCode(99).String(); got != "StatusCode(99)" {
		t.Errorf("StatusCode(99).String() = %q, want StatusCode(99)", got)
	}
	if !errors.Is(StatusCode(99).Err(), ErrUnknown) {
		t.Errorf("unrecognized StatusCode should map to ErrUnknown, got %v", StatusCode(99).Err())
	}
}

func TestCError_FormatAndUnwrap(t *testing.T) {
	err := &CError{
		Op:      "engine_create",
		Code:    StatusNotFound,
		Message: "model file missing",
	}
	if !errors.Is(err, ErrNotFound) {
		t.Errorf("errors.Is(err, ErrNotFound) = false")
	}
	got := err.Error()
	if !strings.Contains(got, "engine_create") || !strings.Contains(got, "model file missing") || !strings.Contains(got, "NotFound") {
		t.Errorf("unexpected Error() string: %q", got)
	}

	noMsg := &CError{Op: "run_prefill", Code: StatusInvalidArgument}
	if !strings.Contains(noMsg.Error(), "run_prefill failed (InvalidArgument)") {
		t.Errorf("unexpected no-message Error(): %q", noMsg.Error())
	}
}

func withStubLastError(t *testing.T, fn func() string) {
	t.Helper()
	prev := readLastErrorLocked
	readLastErrorLocked = fn
	t.Cleanup(func() { readLastErrorLocked = prev })
}

func TestCallStatusHelpers(t *testing.T) {
	t.Run("handle_ok", func(t *testing.T) {
		cb := purego.NewCallback(func(out *uintptr) uintptr {
			*out = 0x12345
			return uintptr(StatusOk)
		})
		var cif ffi.Cif
		if st := ffi.PrepCif(&cif, ffi.DefaultAbi, 1, &ffi.TypeSint32, &ffi.TypePointer); st != ffi.OK {
			t.Fatalf("PrepCif: %v", st)
		}
		fun := ffi.Fun{Addr: cb, Cif: &cif}
		withStubPrep(t, func(string, *ffi.Type, ...*ffi.Type) (ffi.Fun, error) { return fun, nil })

		lf := newLazyFun("stub_create", &ffi.TypeSint32, &ffi.TypePointer)
		got, err := callStatusHandle[Engine](lf, "stub_create")
		if err != nil {
			t.Fatalf("callStatusHandle failed: %v", err)
		}
		if got != Engine(0x12345) {
			t.Errorf("got handle %#x, want 0x12345", got)
		}
	})

	t.Run("scalar_error_with_tls_message", func(t *testing.T) {
		cb := purego.NewCallback(func(out *int32) uintptr {
			return uintptr(StatusInvalidArgument)
		})
		var cif ffi.Cif
		if st := ffi.PrepCif(&cif, ffi.DefaultAbi, 1, &ffi.TypeSint32, &ffi.TypePointer); st != ffi.OK {
			t.Fatalf("PrepCif: %v", st)
		}
		fun := ffi.Fun{Addr: cb, Cif: &cif}
		withStubPrep(t, func(string, *ffi.Type, ...*ffi.Type) (ffi.Fun, error) { return fun, nil })
		withStubLastError(t, func() string { return "bad parameter" })

		lf := newLazyFun("stub_get", &ffi.TypeSint32, &ffi.TypePointer)
		_, err := callStatusScalar[int32](lf, "stub_get")
		if err == nil {
			t.Fatal("expected error")
		}
		if !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("errors.Is(err, ErrInvalidArgument) = false; got %v", err)
		}
		if !strings.Contains(err.Error(), "bad parameter") {
			t.Errorf("error missing TLS message: %q", err.Error())
		}
	})

	t.Run("string_null_is_absent_ok", func(t *testing.T) {
		cb := purego.NewCallback(func(out **byte) uintptr {
			*out = nil
			return uintptr(StatusOk)
		})
		var cif ffi.Cif
		if st := ffi.PrepCif(&cif, ffi.DefaultAbi, 1, &ffi.TypeSint32, &ffi.TypePointer); st != ffi.OK {
			t.Fatalf("PrepCif: %v", st)
		}
		fun := ffi.Fun{Addr: cb, Cif: &cif}
		withStubPrep(t, func(string, *ffi.Type, ...*ffi.Type) (ffi.Fun, error) { return fun, nil })

		lf := newLazyFun("stub_str", &ffi.TypeSint32, &ffi.TypePointer)
		s, ok, err := callStatusString(lf, "stub_str")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if ok || s != "" {
			t.Errorf("got (%q, %v), want (\"\", false)", s, ok)
		}
	})

	t.Run("string_value_copied", func(t *testing.T) {
		cBytes := append([]byte("hello-v1"), 0)
		cb := purego.NewCallback(func(out **byte) uintptr {
			*out = &cBytes[0]
			return uintptr(StatusOk)
		})
		var cif ffi.Cif
		if st := ffi.PrepCif(&cif, ffi.DefaultAbi, 1, &ffi.TypeSint32, &ffi.TypePointer); st != ffi.OK {
			t.Fatalf("PrepCif: %v", st)
		}
		fun := ffi.Fun{Addr: cb, Cif: &cif}
		withStubPrep(t, func(string, *ffi.Type, ...*ffi.Type) (ffi.Fun, error) { return fun, nil })

		lf := newLazyFun("stub_str_val", &ffi.TypeSint32, &ffi.TypePointer)
		s, ok, err := callStatusString(lf, "stub_str_val")
		if err != nil || !ok || s != "hello-v1" {
			t.Fatalf("got (%q, %v, %v), want (\"hello-v1\", true, nil)", s, ok, err)
		}
		_ = unsafe.Pointer(&cBytes[0])
	})
}

func TestCheckAPIVersion(t *testing.T) {
	t.Cleanup(resetAPIVersion)

	t.Run("missing_symbol_rejected", func(t *testing.T) {
		withStubPrep(t, func(string, *ffi.Type, ...*ffi.Type) (ffi.Fun, error) {
			return ffi.Fun{}, errors.New("symbol not found")
		})
		err := checkAPIVersion()
		if err == nil || !strings.Contains(err.Error(), "litert_lm_get_api_version") {
			t.Fatalf("expected missing symbol error, got %v", err)
		}
	})

	t.Run("incompatible_major_rejected", func(t *testing.T) {
		cb := purego.NewCallback(func(maj, min, pat *int32) uintptr {
			*maj, *min, *pat = 2, 0, 0
			return uintptr(StatusOk)
		})
		var cif ffi.Cif
		if st := ffi.PrepCif(&cif, ffi.DefaultAbi, 3, &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer); st != ffi.OK {
			t.Fatalf("PrepCif: %v", st)
		}
		withStubPrep(t, func(string, *ffi.Type, ...*ffi.Type) (ffi.Fun, error) {
			return ffi.Fun{Addr: cb, Cif: &cif}, nil
		})
		err := checkAPIVersion()
		if err == nil || !strings.Contains(err.Error(), "unsupported C API version 2.0.0") {
			t.Fatalf("expected major version error, got %v", err)
		}
	})

	t.Run("v1_0_0_accepted_and_require_1_1_blocked", func(t *testing.T) {
		cb := purego.NewCallback(func(maj, min, pat *int32) uintptr {
			*maj, *min, *pat = 1, 0, 0
			return uintptr(StatusOk)
		})
		var cif ffi.Cif
		if st := ffi.PrepCif(&cif, ffi.DefaultAbi, 3, &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer); st != ffi.OK {
			t.Fatalf("PrepCif: %v", st)
		}
		withStubPrep(t, func(string, *ffi.Type, ...*ffi.Type) (ffi.Fun, error) {
			return ffi.Fun{Addr: cb, Cif: &cif}, nil
		})
		if err := checkAPIVersion(); err != nil {
			t.Fatalf("checkAPIVersion(1.0.0) failed: %v", err)
		}
		maj, min, pat, err := APIVersion()
		if err != nil || maj != 1 || min != 0 || pat != 0 {
			t.Fatalf("APIVersion() = (%d,%d,%d,%v), want (1,0,0,nil)", maj, min, pat, err)
		}
		if err := requireAPIVersion(1, 0); err != nil {
			t.Errorf("requireAPIVersion(1, 0) unexpected error: %v", err)
		}
		if err := requireAPIVersion(1, 1); !errors.Is(err, ErrUnimplemented) {
			t.Errorf("requireAPIVersion(1, 1) = %v, want ErrUnimplemented", err)
		}
	})

	t.Run("v1_1_0_passes_require_1_1", func(t *testing.T) {
		cb := purego.NewCallback(func(maj, min, pat *int32) uintptr {
			*maj, *min, *pat = 1, 1, 0
			return uintptr(StatusOk)
		})
		var cif ffi.Cif
		if st := ffi.PrepCif(&cif, ffi.DefaultAbi, 3, &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer); st != ffi.OK {
			t.Fatalf("PrepCif: %v", st)
		}
		withStubPrep(t, func(string, *ffi.Type, ...*ffi.Type) (ffi.Fun, error) {
			return ffi.Fun{Addr: cb, Cif: &cif}, nil
		})
		if err := checkAPIVersion(); err != nil {
			t.Fatalf("checkAPIVersion(1.1.0) failed: %v", err)
		}
		if err := requireAPIVersion(1, 1); err != nil {
			t.Errorf("requireAPIVersion(1, 1) failed: %v", err)
		}
	})
}

func BenchmarkCallStatusScalar(b *testing.B) {
	cb := purego.NewCallback(func(_ uintptr, out *uint64) uintptr {
		*out = 42
		return uintptr(StatusOk)
	})
	var cif ffi.Cif
	if st := ffi.PrepCif(&cif, ffi.DefaultAbi, 2, &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer); st != ffi.OK {
		b.Fatalf("PrepCif: %v", st)
	}
	lf := &lazyFun{
		name: "bench_scalar",
		ret:  &ffi.TypeSint32,
		args: []*ffi.Type{&ffi.TypePointer, &ffi.TypePointer},
		fun:  ffi.Fun{Addr: cb, Cif: &cif},
	}
	lf.once.Do(func() {})

	handle := uintptr(1)
	b.ReportAllocs()
	for b.Loop() {
		v, err := callStatusScalar[uint64](lf, "bench_scalar", unsafe.Pointer(&handle))
		if err != nil || v != 42 {
			b.Fatalf("got (%d, %v)", v, err)
		}
	}
}
