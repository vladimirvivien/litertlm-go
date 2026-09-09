package litertlm

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/jupiterrider/ffi"
)

// prepSymbol is the resolver lazyFun delegates to. Defaults to rt.prepSymbol;
// tests swap it via withStubPrep without touching rt.libHandle (whose zero
// value behaves platform-dependently — Linux glibc treats Dlsym(0, name) as
// RTLD_DEFAULT).
var prepSymbol = func(name string, ret *ffi.Type, args ...*ffi.Type) (ffi.Fun, error) {
	return rt.prepSymbol(name, ret, args...)
}

// lazyFun resolves a single C symbol on first call and caches the prepared
// ffi.Fun (or resolution error). Missing symbols surface as *CError with code
// StatusUnimplemented from callErr rather than panicking.
type lazyFun struct {
	name string
	ret  *ffi.Type
	args []*ffi.Type

	once sync.Once
	fun  ffi.Fun
	err  error
}

// newLazyFun records the symbol's name and FFI signature. Resolution
// is deferred until the first call.
func newLazyFun(name string, ret *ffi.Type, args ...*ffi.Type) *lazyFun {
	return &lazyFun{name: name, ret: ret, args: args}
}

// callErr resolves the symbol on first invocation, then forwards to
// ffi.Fun.Call. Missing symbols return a *CError wrapping ErrUnimplemented.
//
// libffi widens sub-register integral return values to a full ffi_arg
// (8 bytes on 64-bit) when writing through ret, so sub-register returns are
// routed through a register-width scratch word before copying back l.ret.Size
// bytes.
func (l *lazyFun) callErr(ret any, args ...any) error {
	l.once.Do(func() {
		l.fun, l.err = prepSymbol(l.name, l.ret, l.args...)
	})
	if l.err != nil {
		return &CError{
			Op:      l.name,
			Code:    StatusUnimplemented,
			Message: fmt.Sprintf("missing C symbol %q in loaded library (refresh prebuilt LiteRT-LM libs)", l.name),
			Cause:   l.err,
		}
	}

	dst, isPtr := ret.(unsafe.Pointer)
	if !isPtr || l.ret == nil || l.ret.Size == 0 || l.ret.Size >= 8 {
		l.fun.Call(ret, args...)
		return nil
	}
	var wide ffi.Arg
	l.fun.Call(unsafe.Pointer(&wide), args...)
	n := int(l.ret.Size)
	copy(unsafe.Slice((*byte)(dst), n), unsafe.Slice((*byte)(unsafe.Pointer(&wide)), n))
	return nil
}

// callVoid invokes a C entry point whose error return is intentionally ignored
// (such as idempotent handle destructors).
func (l *lazyFun) callVoid(args ...any) {
	_ = l.callErr(nil, args...)
}
