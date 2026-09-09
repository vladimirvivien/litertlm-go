package litertlm

import (
	"errors"
	"fmt"
	"runtime"
	"strconv"
	"unsafe"

	"github.com/jupiterrider/ffi"

	"github.com/vladimirvivien/litertlm-go/internal/cstr"
)

// StatusCode represents canonical status codes returned by LiteRT-LM C API 1.0.0+
// functions. Values mirror `LiteRtLmStatusCode` in c/error_reporter.h (`absl::StatusCode`).
type StatusCode int32

const (
	StatusOk                 StatusCode = 0
	StatusCancelled          StatusCode = 1
	StatusUnknown            StatusCode = 2
	StatusInvalidArgument    StatusCode = 3
	StatusDeadlineExceeded   StatusCode = 4
	StatusNotFound           StatusCode = 5
	StatusAlreadyExists      StatusCode = 6
	StatusPermissionDenied   StatusCode = 7
	StatusResourceExhausted  StatusCode = 8
	StatusFailedPrecondition StatusCode = 9
	StatusAborted            StatusCode = 10
	StatusOutOfRange         StatusCode = 11
	StatusUnimplemented      StatusCode = 12
	StatusInternal           StatusCode = 13
	StatusUnavailable        StatusCode = 14
	StatusDataLoss           StatusCode = 15
	StatusUnauthenticated    StatusCode = 16
)

// Sentinel errors matching canonical C API status codes for use with errors.Is.
var (
	ErrCancelled          = errors.New("cancelled")
	ErrUnknown            = errors.New("unknown error")
	ErrInvalidArgument    = errors.New("invalid argument")
	ErrDeadlineExceeded   = errors.New("deadline exceeded")
	ErrNotFound           = errors.New("not found")
	ErrAlreadyExists      = errors.New("already exists")
	ErrPermissionDenied   = errors.New("permission denied")
	ErrResourceExhausted  = errors.New("resource exhausted")
	ErrFailedPrecondition = errors.New("failed precondition")
	ErrAborted            = errors.New("aborted")
	ErrOutOfRange         = errors.New("out of range")
	ErrUnimplemented      = errors.New("unimplemented")
	ErrInternal           = errors.New("internal error")
	ErrUnavailable        = errors.New("unavailable")
	ErrDataLoss           = errors.New("data loss")
	ErrUnauthenticated    = errors.New("unauthenticated")
)

var statusCodeNames = [...]string{
	StatusOk:                 "OK",
	StatusCancelled:          "Cancelled",
	StatusUnknown:            "Unknown",
	StatusInvalidArgument:    "InvalidArgument",
	StatusDeadlineExceeded:   "DeadlineExceeded",
	StatusNotFound:           "NotFound",
	StatusAlreadyExists:      "AlreadyExists",
	StatusPermissionDenied:   "PermissionDenied",
	StatusResourceExhausted:  "ResourceExhausted",
	StatusFailedPrecondition: "FailedPrecondition",
	StatusAborted:            "Aborted",
	StatusOutOfRange:         "OutOfRange",
	StatusUnimplemented:      "Unimplemented",
	StatusInternal:           "Internal",
	StatusUnavailable:        "Unavailable",
	StatusDataLoss:           "DataLoss",
	StatusUnauthenticated:    "Unauthenticated",
}

var statusSentinels = [...]error{
	StatusOk:                 nil,
	StatusCancelled:          ErrCancelled,
	StatusUnknown:            ErrUnknown,
	StatusInvalidArgument:    ErrInvalidArgument,
	StatusDeadlineExceeded:   ErrDeadlineExceeded,
	StatusNotFound:           ErrNotFound,
	StatusAlreadyExists:      ErrAlreadyExists,
	StatusPermissionDenied:   ErrPermissionDenied,
	StatusResourceExhausted:  ErrResourceExhausted,
	StatusFailedPrecondition: ErrFailedPrecondition,
	StatusAborted:            ErrAborted,
	StatusOutOfRange:         ErrOutOfRange,
	StatusUnimplemented:      ErrUnimplemented,
	StatusInternal:           ErrInternal,
	StatusUnavailable:        ErrUnavailable,
	StatusDataLoss:           ErrDataLoss,
	StatusUnauthenticated:    ErrUnauthenticated,
}

// String returns the canonical name of the status code.
func (s StatusCode) String() string {
	if s >= 0 && int(s) < len(statusCodeNames) {
		return statusCodeNames[s]
	}
	return "StatusCode(" + strconv.Itoa(int(s)) + ")"
}

// Err maps the status code to its corresponding sentinel error.
// StatusOk returns nil; unrecognized values map to ErrUnknown per c/error_reporter.h.
func (s StatusCode) Err() error {
	if s == StatusOk {
		return nil
	}
	if s > 0 && int(s) < len(statusSentinels) {
		return statusSentinels[s]
	}
	return ErrUnknown
}

// CError represents a non-OK status or symbol-resolution error from the
// LiteRT-LM C API along with any thread-local diagnostic message.
type CError struct {
	Op      string
	Code    StatusCode
	Message string
	Cause   error
}

// Error formats the C API failure with operation, message, and status code.
func (e *CError) Error() string {
	if e.Message != "" && e.Op != "" {
		return fmt.Sprintf("litertlm: %s: %s (%s)", e.Op, e.Message, e.Code)
	}
	if e.Message != "" {
		return fmt.Sprintf("litertlm: %s (%s)", e.Message, e.Code)
	}
	if e.Op != "" {
		return fmt.Sprintf("litertlm: %s failed (%s)", e.Op, e.Code)
	}
	return fmt.Sprintf("litertlm: error (%s)", e.Code)
}

// Unwrap returns the sentinel error for e.Code and any underlying Cause so
// both match via errors.Is and errors.As.
func (e *CError) Unwrap() []error {
	if e == nil {
		return nil
	}
	var errs []error
	if sentinel := e.Code.Err(); sentinel != nil {
		errs = append(errs, sentinel)
	}
	if e.Cause != nil {
		errs = append(errs, e.Cause)
	}
	return errs
}

// APIVersion returns the semantic version (major, minor, patch) reported by
// the loaded LiteRT-LM C API library.
func APIVersion() (major, minor, patch int, err error) {
	if maj, minVer, pat, ok := rt.apiVersion(); ok {
		return maj, minVer, pat, nil
	}
	if !rt.isLoaded() {
		return 0, 0, 0, fmt.Errorf("litertlm: APIVersion: library not loaded")
	}
	return queryAPIVersion()
}

// ClearLastError resets the calling thread's thread-local C error message.
func ClearLastError() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	clearLastErrorFunc.callVoid()
}

func checkAPIVersion() error {
	maj, minorVer, pat, err := queryAPIVersion()
	if err != nil {
		rt.resetAPIVersion()
		return err
	}
	if maj != 1 {
		rt.resetAPIVersion()
		return fmt.Errorf("litertlm: unsupported C API version %d.%d.%d (expected major version 1)", maj, minorVer, pat)
	}
	rt.setAPIVersion(maj, minorVer, pat)
	return nil
}

func resetAPIVersion() {
	rt.resetAPIVersion()
}

func requireAPIVersion(minMajor, minMinor int) error {
	maj, minorVer, pat, ok := rt.apiVersion()
	if !ok {
		return nil
	}
	if maj != minMajor || minorVer < minMinor {
		return &CError{
			Op:      "version_check",
			Code:    StatusUnimplemented,
			Message: fmt.Sprintf("requires LiteRT-LM C API %d.%d+, loaded library is %d.%d.%d", minMajor, minMinor, maj, minorVer, pat),
		}
	}
	return nil
}

func queryAPIVersion() (int, int, int, error) {
	fun, err := prepSymbol(
		"litert_lm_get_api_version",
		&ffi.TypeSint32,
		&ffi.TypePointer,
		&ffi.TypePointer,
		&ffi.TypePointer,
	)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("litertlm: loaded library does not export litert_lm_get_api_version (libraries older than LiteRT-LM v0.19.0 / C API 1.0.0 are unsupported): %w", err)
	}
	lf := &lazyFun{
		name: "litert_lm_get_api_version",
		ret:  &ffi.TypeSint32,
		args: []*ffi.Type{&ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer},
		fun:  fun,
	}
	lf.once.Do(func() {})

	var maj, minorVer, pat int32
	majPtr, minorPtr, patPtr := &maj, &minorVer, &pat
	if err = callStatus(lf, "get_api_version",
		unsafe.Pointer(&majPtr),
		unsafe.Pointer(&minorPtr),
		unsafe.Pointer(&patPtr),
	); err != nil {
		runtime.KeepAlive(majPtr)
		runtime.KeepAlive(minorPtr)
		runtime.KeepAlive(patPtr)
		return 0, 0, 0, err
	}
	runtime.KeepAlive(majPtr)
	runtime.KeepAlive(minorPtr)
	runtime.KeepAlive(patPtr)
	return int(maj), int(minorVer), int(pat), nil
}

// readLastErrorLocked reads the thread-local error message recorded by the
// failing C API call on the same OS thread. Replaced in unit tests.
var readLastErrorLocked = func() string {
	var ptr *byte
	if err := getLastErrorMessageFunc.callErr(unsafe.Pointer(&ptr)); err != nil || ptr == nil {
		return ""
	}
	return cstr.BytePtrToString(ptr)
}

func callStatusArgs(f *lazyFun, op string, args []unsafe.Pointer, out unsafe.Pointer) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	total := len(args)
	if out != nil {
		total++
	}
	var stack [8]any
	var anyArgs []any
	if total <= len(stack) {
		anyArgs = stack[:total]
	} else {
		anyArgs = make([]any, total)
	}
	for i, a := range args {
		anyArgs[i] = a
	}
	if out != nil {
		anyArgs[len(args)] = out
	}

	var status int32
	if err := f.callErr(unsafe.Pointer(&status), anyArgs...); err != nil {
		return err
	}
	code := StatusCode(status)
	if code == StatusOk {
		return nil
	}
	return &CError{
		Op:      op,
		Code:    code,
		Message: readLastErrorLocked(),
	}
}

// callStatus invokes a C API function returning LiteRtLmStatusCode. It locks
// the goroutine to its OS thread so any error message can be read safely from
// C++ thread-local storage on failure.
func callStatus(f *lazyFun, op string, args ...unsafe.Pointer) error {
	return callStatusArgs(f, op, args, nil)
}

// callStatusHandle invokes a C API constructor or handle getter whose trailing
// out-parameter is an opaque handle pointer `H**`.
func callStatusHandle[H cHandle](f *lazyFun, op string, args ...unsafe.Pointer) (H, error) {
	return callStatusScalar[H](f, op, args...)
}

// callStatusScalar invokes a C API getter whose trailing out-parameter is `T*`.
func callStatusScalar[T any](f *lazyFun, op string, args ...unsafe.Pointer) (T, error) {
	var v T
	outPtr := &v
	err := callStatusArgs(f, op, args, unsafe.Pointer(&outPtr))
	runtime.KeepAlive(outPtr)
	if err != nil {
		var zero T
		return zero, err
	}
	return v, nil
}

// callStatusBool invokes a C API predicate whose trailing out-parameter is `bool*`.
func callStatusBool(f *lazyFun, op string, args ...unsafe.Pointer) (bool, error) {
	v, err := callStatusScalar[uint8](f, op, args...)
	if err != nil {
		return false, err
	}
	return v != 0, nil
}

// callStatusString invokes a C API getter whose trailing out-parameter is `const char**`.
// ok is false when the C API succeeded with a NULL output pointer (legitimately absent value).
func callStatusString(f *lazyFun, op string, args ...unsafe.Pointer) (val string, ok bool, err error) {
	ptr, err := callStatusScalar[*byte](f, op, args...)
	if err != nil {
		return "", false, err
	}
	if ptr == nil {
		return "", false, nil
	}
	return cstr.BytePtrToString(ptr), true, nil
}
