package litertlm

import (
	"fmt"
	"unsafe"

	"github.com/vladimirvivien/litertlm-go/pkg/utils"
)

// bytePtrOrNil returns a null-terminated byte pointer for s, or nil for
// the empty string. Several C entry points accept NULL for "unset".
func bytePtrOrNil(s string) (*byte, error) {
	if s == "" {
		return nil, nil
	}
	return utils.BytePtrFromString(s)
}

// callForHandle invokes a C constructor that writes a heap-allocated
// pointer (cast to a uintptr-sized handle H) into its first out-parameter,
// and returns NULL on failure. It collapses the declare/Call/null-check/
// wrap-error block that every constructor in this package would otherwise
// repeat. sym is the bare C symbol name; the formatted error reads
// `litertlm: <sym> failed`.
//
// lazyFun.Call's signature is `(ret any, args ...any)` so we widen the
// typed unsafe.Pointer args here rather than at every callsite.
func callForHandle[H ~uintptr](f *lazyFun, sym string, args ...unsafe.Pointer) (H, error) {
	var h H
	anyArgs := make([]any, len(args))
	for i, a := range args {
		anyArgs[i] = a
	}
	f.Call(unsafe.Pointer(&h), anyArgs...)
	if h == 0 {
		return 0, fmt.Errorf("litertlm: %s failed", sym)
	}
	return h, nil
}
