package litertlm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"strings"
	"testing"
	"unsafe"

	"github.com/jupiterrider/ffi"
)

type stubNetErr struct {
	timeout bool
}

func (e stubNetErr) Error() string   { return "network failure" }
func (e stubNetErr) Timeout() bool   { return e.timeout }
func (e stubNetErr) Temporary() bool { return true }

type stubHTTPErr int

func (s stubHTTPErr) Error() string       { return fmt.Sprintf("http %d", s) }
func (s stubHTTPErr) HTTPStatusCode() int { return int(s) }

func withStubAPIVersion(t *testing.T, maj, minVer, pat int) {
	t.Helper()
	prevMaj, prevMin, prevPat, prevOK := rt.apiVersion()
	rt.setAPIVersion(maj, minVer, pat)
	t.Cleanup(func() {
		if prevOK {
			rt.setAPIVersion(prevMaj, prevMin, prevPat)
		} else {
			rt.resetAPIVersion()
		}
	})
}

func withStubRangeWriterHooks(t *testing.T, writeFn func(ModelSyncRangeWriter, unsafe.Pointer, uint64) error, errFn func(ModelSyncRangeWriter, *byte) error) {
	t.Helper()
	prevW, prevE := modelSyncWriteCheck, modelSyncErrCheck
	if writeFn != nil {
		modelSyncWriteCheck = writeFn
	}
	if errFn != nil {
		modelSyncErrCheck = errFn
	}
	t.Cleanup(func() {
		modelSyncWriteCheck = prevW
		modelSyncErrCheck = prevE
	})
}

func TestModelSync_ZeroHandlesAndValidation(t *testing.T) {
	var opt ModelSyncOptionsHandle
	opt.Delete()
	if err := opt.SetFullDownloadThreshold(0.5); err == nil {
		t.Error("expected error on zero options SetFullDownloadThreshold")
	}
	if err := ModelSyncOptionsHandle(1).SetFullDownloadThreshold(-0.1); err == nil {
		t.Error("expected error on negative threshold")
	}
	if err := ModelSyncOptionsHandle(1).SetFullDownloadThreshold(1.1); err == nil {
		t.Error("expected error on >1 threshold")
	}
	if err := ModelSyncOptionsHandle(1).SetFullDownloadThreshold(math.NaN()); err == nil {
		t.Error("expected error on NaN threshold")
	}
	if err := opt.SetClean(true); err == nil {
		t.Error("expected error on zero options SetClean")
	}
	if err := opt.SetCheckDiskSpace(true); err == nil {
		t.Error("expected error on zero options SetCheckDiskSpace")
	}
	if err := opt.SetAllowOfflineFallback(true); err == nil {
		t.Error("expected error on zero options SetAllowOfflineFallback")
	}

	var plan ModelSyncPlanHandle
	plan.Delete()
	if _, err := plan.IsUpToDate(); err == nil {
		t.Error("expected error on zero plan IsUpToDate")
	}
	if _, err := plan.BytesToDownload(); err == nil {
		t.Error("expected error on zero plan BytesToDownload")
	}
	if _, err := plan.BytesToReuse(); err == nil {
		t.Error("expected error on zero plan BytesToReuse")
	}
	if _, err := plan.TotalRemoteSize(); err == nil {
		t.Error("expected error on zero plan TotalRemoteSize")
	}
	if _, err := plan.Format(); err == nil {
		t.Error("expected error on zero plan Format")
	}
	if _, err := plan.Snapshot(); err == nil {
		t.Error("expected error on zero plan Snapshot")
	}

	var res ModelSyncResultHandle
	res.Delete()
	if _, err := res.CachedPath(); err == nil {
		t.Error("expected error on zero result CachedPath")
	}

	var w ModelSyncRangeWriter
	if _, err := w.Write([]byte("hi")); err == nil {
		t.Error("expected error on zero writer Write")
	}
	if n, err := ModelSyncRangeWriter(1).Write(nil); err != nil || n != 0 {
		t.Errorf("empty Write = (%d, %v), want (0, nil)", n, err)
	}
	if err := w.SetError("boom"); err == nil {
		t.Error("expected error on zero writer SetError")
	}

	dummyFetcher := RangeFetcherFunc(func(context.Context, uint64, uint64, io.Writer) error { return nil })
	ctx := context.Background()
	if _, err := NewModelSyncPlanFromFile(ctx, "", nil, 0); err == nil {
		t.Error("expected error on nil fetcher in NewModelSyncPlanFromFile")
	}
	if err := ModelSyncUpdate(ctx, "", "", dummyFetcher, 0); err == nil {
		t.Error("expected error on empty output path in ModelSyncUpdate")
	}
	if _, err := ModelSyncCacheSync(ctx, "", "m", dummyFetcher, 0); err == nil {
		t.Error("expected error on empty cache dir in ModelSyncCacheSync")
	}
	if _, err := ModelSyncCacheSync(ctx, t.TempDir(), "", dummyFetcher, 0); err == nil {
		t.Error("expected error on empty model ID in ModelSyncCacheSync")
	}
	if _, err := PlanModelSync(ctx, "", dummyFetcher, WithFullDownloadThreshold(2.0)); err == nil {
		t.Error("expected error on invalid WithFullDownloadThreshold")
	}

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := PlanModelSync(canceled, "", dummyFetcher); !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}

func TestModelSync_VersionGating(t *testing.T) {
	withStubAPIVersion(t, 1, 0, 0)
	dummyFetcher := RangeFetcherFunc(func(context.Context, uint64, uint64, io.Writer) error { return nil })
	ctx := context.Background()

	assertUnimplemented := func(name string, err error) {
		t.Helper()
		if !errors.Is(err, ErrUnimplemented) {
			t.Errorf("%s error = %v, want ErrUnimplemented", name, err)
		}
		if _, ok := errors.AsType[*CError](err); !ok {
			t.Errorf("%s error type = %T, want *CError", name, err)
		}
	}

	_, optErr := NewModelSyncOptions()
	assertUnimplemented("NewModelSyncOptions", optErr)

	_, planErr := PlanModelSync(ctx, "", dummyFetcher)
	assertUnimplemented("PlanModelSync", planErr)

	syncErr := SyncModel(ctx, "", "out.litertlm", dummyFetcher)
	assertUnimplemented("SyncModel", syncErr)

	_, cacheErr := SyncModelCache(ctx, t.TempDir(), "gemma", dummyFetcher)
	assertUnimplemented("SyncModelCache", cacheErr)
}

func TestModelSync_MapRangeFetchError(t *testing.T) {
	var zeroNet net.Error = stubNetErr{timeout: false}
	var timeoutNet net.Error = stubNetErr{timeout: true}

	tests := []struct {
		name string
		err  error
		want StatusCode
	}{
		{"nil", nil, StatusOk},
		{"canceled", context.Canceled, StatusCancelled},
		{"deadline", context.DeadlineExceeded, StatusDeadlineExceeded},
		{"c_error", &CError{Op: "custom", Code: StatusPermissionDenied}, StatusPermissionDenied},
		{"http_401", stubHTTPErr(http.StatusUnauthorized), StatusUnauthenticated},
		{"http_403", stubHTTPErr(http.StatusForbidden), StatusPermissionDenied},
		{"http_404", stubHTTPErr(http.StatusNotFound), StatusNotFound},
		{"http_416", stubHTTPErr(http.StatusRequestedRangeNotSatisfiable), StatusOutOfRange},
		{"http_429", stubHTTPErr(http.StatusTooManyRequests), StatusUnavailable},
		{"http_503", stubHTTPErr(http.StatusServiceUnavailable), StatusUnavailable},
		{"http_400", stubHTTPErr(http.StatusBadRequest), StatusInvalidArgument},
		{"net_timeout", timeoutNet, StatusDeadlineExceeded},
		{"net_unavailable", zeroNet, StatusUnavailable},
		{"sentinel_unavailable", fmt.Errorf("wrap: %w", ErrUnavailable), StatusUnavailable},
		{"sentinel_out_of_range", ErrOutOfRange, StatusOutOfRange},
		{"sentinel_data_loss", ErrDataLoss, StatusDataLoss},
		{"generic", errors.New("boom"), StatusInternal},
	}

	for _, tc := range tests {
		if got := mapRangeFetchError(tc.err); got != tc.want {
			t.Errorf("%s: mapRangeFetchError = %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestModelSync_CallbackTrampoline(t *testing.T) {
	ensureModelSyncTrampoline()
	if modelSyncTrampAddr == 0 {
		t.Fatal("expected nonzero purego callback address for modelSyncTrampoline")
	}

	var wrote []byte
	var recordedErr string
	withStubRangeWriterHooks(t,
		func(_ ModelSyncRangeWriter, dataPtr unsafe.Pointer, size uint64) error {
			wrote = append(wrote, unsafe.Slice((*byte)(dataPtr), size)...)
			return nil
		},
		func(_ ModelSyncRangeWriter, msgPtr *byte) error {
			if msgPtr != nil {
				// Reconstruct string cleanly without relying on C runtime.
				for p := msgPtr; *p != 0; p = (*byte)(unsafe.Add(unsafe.Pointer(p), 1)) {
					recordedErr += string(*p)
				}
			}
			return nil
		},
	)

	var cif ffi.Cif
	if st := ffi.PrepCif(
		&cif, ffi.DefaultAbi, 4, &ffi.TypePointer,
		&ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer,
	); st != ffi.OK {
		t.Fatalf("PrepCif: %v", st)
	}
	fun := ffi.Fun{Addr: modelSyncTrampAddr, Cif: &cif}

	callTramp := func(off, length, writer, cookie uintptr) StatusCode {
		var ret uintptr
		fun.Call(unsafe.Pointer(&ret),
			unsafe.Pointer(&off),
			unsafe.Pointer(&length),
			unsafe.Pointer(&writer),
			unsafe.Pointer(&cookie),
		)
		return StatusCode(ret)
	}

	// Missing cookie or nil writer returns StatusInvalidArgument.
	if st := callTramp(0, 4, 0, 99999); st != StatusInvalidArgument {
		t.Errorf("unknown cookie status = %v, want InvalidArgument", st)
	}

	// Successful callback write via real libffi trampoline dispatch.
	okCookie, _, cleanupOK := registerSyncFetcher(context.Background(), RangeFetcherFunc(
		func(_ context.Context, offset, length uint64, w io.Writer) error {
			if offset != 8 || length != 5 {
				return fmt.Errorf("unexpected range (%d, %d)", offset, length)
			}
			_, err := w.Write([]byte("hello"))
			return err
		},
	))
	st := callTramp(8, 5, 100, okCookie)
	cleanupOK()
	if st != StatusOk || string(wrote) != "hello" {
		t.Errorf("success callback got (status=%v, wrote=%q), want (OK, \"hello\")", st, wrote)
	}

	// Error mapping + diagnostic message propagation through trampoline.
	errCookie, _, cleanupErr := registerSyncFetcher(context.Background(), RangeFetcherFunc(
		func(context.Context, uint64, uint64, io.Writer) error {
			return stubHTTPErr(http.StatusServiceUnavailable)
		},
	))
	st = callTramp(0, 4, 100, errCookie)
	cleanupErr()
	if st != StatusUnavailable || !strings.Contains(recordedErr, "http 503") {
		t.Errorf("error callback got (status=%v, err=%q), want (Unavailable, contains http 503)", st, recordedErr)
	}
}
