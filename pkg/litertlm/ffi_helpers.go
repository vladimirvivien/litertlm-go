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
