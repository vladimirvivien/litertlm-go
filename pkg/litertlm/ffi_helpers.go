package litertlm

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/vladimirvivien/litertlm-go/internal/cstr"
)

// cHandle matches any opaque C handle type represented as a uintptr.
type cHandle interface {
	~uintptr
}

// cScalar matches fixed-width C scalar arguments passed by pointer into FFI.
type cScalar interface {
	int32 | uint8 | float32 | float64 | uint64
}

func deleteHandle[H cHandle](h H, f *lazyFun) {
	if h == 0 {
		return
	}
	f.callVoid(unsafe.Pointer(&h))
}

func setHandleScalar[H cHandle, V cScalar](h H, handleName string, f *lazyFun, op string, v V) error {
	if h == 0 {
		return fmt.Errorf("litertlm: %s: invalid %s", op, handleName)
	}
	val := v
	return callStatus(f, op, unsafe.Pointer(&h), unsafe.Pointer(&val))
}

func setHandleBool[H cHandle](h H, handleName string, f *lazyFun, op string, on bool) error {
	var v uint8
	if on {
		v = 1
	}
	return setHandleScalar(h, handleName, f, op, v)
}

func setHandleString[H cHandle](h H, handleName string, f *lazyFun, op, s string) error {
	if h == 0 {
		return fmt.Errorf("litertlm: %s: invalid %s", op, handleName)
	}
	ptr, err := cstr.BytePtrFromString(s)
	if err != nil {
		return err
	}
	err = callStatus(f, op, unsafe.Pointer(&h), unsafe.Pointer(&ptr))
	runtime.KeepAlive(ptr)
	return err
}

func setHandleConfig[H cHandle, C cHandle](h H, handleName string, f *lazyFun, op string, cfg C) error {
	if h == 0 {
		return fmt.Errorf("litertlm: %s: invalid %s", op, handleName)
	}
	err := callStatus(f, op, unsafe.Pointer(&h), unsafe.Pointer(&cfg))
	runtime.KeepAlive(cfg)
	return err
}

func setHandleStringSlice[H cHandle](h H, handleName string, f *lazyFun, op string, vals []string) error {
	if h == 0 {
		return fmt.Errorf("litertlm: %s: invalid %s", op, handleName)
	}
	if len(vals) == 0 {
		return fmt.Errorf("litertlm: %s: empty slice", op)
	}
	ptrs := make([]*byte, len(vals))
	for i, s := range vals {
		p, err := cstr.BytePtrFromString(s)
		if err != nil {
			return err
		}
		ptrs[i] = p
	}
	arrPtr := unsafe.Pointer(&ptrs[0])
	n := uint64(len(ptrs))
	err := callStatus(f, op, unsafe.Pointer(&h), unsafe.Pointer(&arrPtr), unsafe.Pointer(&n))
	runtime.KeepAlive(ptrs)
	return err
}

func withTempHandle[H cHandle](build func() (H, error), del func(H), use func(H) error) error {
	h, err := build()
	if err != nil {
		return err
	}
	defer del(h)
	return use(h)
}

func setOptionalConfig[H cHandle, C any](
	o OptionalArgs,
	op string,
	f *lazyFun,
	cfg *C,
	build func(C) (H, error),
	del func(H),
) error {
	if cfg == nil {
		return setHandleConfig[OptionalArgs, H](o, "optional args", f, op, 0)
	}
	return withTempHandle(
		func() (H, error) { return build(*cfg) },
		del,
		func(h H) error { return setHandleConfig(o, "optional args", f, op, h) },
	)
}

func setHandleInt32Slice[H cHandle](h H, handleName string, f *lazyFun, op string, vals []int) error {
	if h == 0 {
		return fmt.Errorf("litertlm: %s: invalid %s", op, handleName)
	}
	if len(vals) == 0 {
		return nil
	}
	buf := make([]int32, len(vals))
	for i, v := range vals {
		buf[i] = int32(v)
	}
	arrPtr := unsafe.Pointer(&buf[0])
	n := uint64(len(buf))
	err := callStatus(f, op, unsafe.Pointer(&h), unsafe.Pointer(&arrPtr), unsafe.Pointer(&n))
	runtime.KeepAlive(buf)
	return err
}

func getHandleScalar[H cHandle, T any](h H, f *lazyFun, op string, fallback T) T {
	if h == 0 {
		return fallback
	}
	v, err := callStatusScalar[T](f, op, unsafe.Pointer(&h))
	if err != nil {
		return fallback
	}
	return v
}

func getHandleScalarAt[H cHandle, T any](h H, i int, f *lazyFun, op string, fallback T) T {
	if h == 0 || i < 0 {
		return fallback
	}
	idx := uint64(i)
	v, err := callStatusScalar[T](f, op, unsafe.Pointer(&h), unsafe.Pointer(&idx))
	if err != nil {
		return fallback
	}
	return v
}

func getHandleBool[H cHandle](h H, f *lazyFun, op string) bool {
	if h == 0 {
		return false
	}
	v, err := callStatusBool(f, op, unsafe.Pointer(&h))
	return err == nil && v
}

func getHandleScalarErr[H cHandle, T any](h H, handleName string, f *lazyFun, op string) (T, error) {
	var zero T
	if h == 0 {
		return zero, fmt.Errorf("litertlm: %s: invalid %s", op, handleName)
	}
	return callStatusScalar[T](f, op, unsafe.Pointer(&h))
}

func getHandleBoolErr[H cHandle](h H, handleName string, f *lazyFun, op string) (bool, error) {
	if h == 0 {
		return false, fmt.Errorf("litertlm: %s: invalid %s", op, handleName)
	}
	return callStatusBool(f, op, unsafe.Pointer(&h))
}

func getHandleStringErr[H cHandle](h H, handleName string, f *lazyFun, op string) (string, error) {
	if h == 0 {
		return "", fmt.Errorf("litertlm: %s: invalid %s", op, handleName)
	}
	s, _, err := callStatusString(f, op, unsafe.Pointer(&h))
	if err != nil {
		return "", err
	}
	return s, nil
}

// queryCBuffer runs the two-step C size-query-then-fill convention: call once
// with (nil, 0) to obtain the total element count, allocate a Go slice of that
// size, and call again to populate it.
func queryCBuffer[T any](call func(buf *T, maxSize uint64) (uint64, error)) ([]T, error) {
	count, err := call(nil, 0)
	if err != nil {
		return nil, err
	}
	if count == 0 {
		return nil, nil
	}
	buf := make([]T, count)
	got, err := call(&buf[0], count)
	runtime.KeepAlive(buf)
	if err != nil {
		return nil, err
	}
	if got < count {
		buf = buf[:got]
	}
	return buf, nil
}

func copyCSlice[T any](ptr *T, n uint64) []T {
	if ptr == nil || n == 0 {
		return []T{}
	}
	out := make([]T, n)
	copy(out, unsafe.Slice(ptr, n))
	return out
}

func handlePtrs[H cHandle](handles []H) []uintptr {
	ptrs := make([]uintptr, len(handles))
	for i, h := range handles {
		ptrs[i] = uintptr(h)
	}
	return ptrs
}
