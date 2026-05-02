package litertlm

import (
	"fmt"
	"sync"

	"github.com/jupiterrider/ffi"
)

// libHandle holds the main C-API library opened by Load. lazyFun calls
// resolve symbols against it on first use. Set under loadMu inside
// Load; subsequent reads happen-after that write because no binding
// call site is reachable until Load returns.
var libHandle ffi.Lib

// lazyFun resolves a single C symbol against libHandle on first Call.
// On resolution failure the call panics with a message that names the
// missing symbol — the actionable failure for users staging an older
// LiteRT-LM library that predates a binding the Go side knows about.
//
// Storing one error per symbol (rather than a central registry) keeps
// the surface tiny: bindings.go just declares one *lazyFun per C
// entry point and call sites use the same .Call(ret, args...) shape
// they used when these were ffi.Fun values.
type lazyFun struct {
	name string
	ret  *ffi.Type
	args []*ffi.Type

	once sync.Once
	fun  ffi.Fun
	err  error
}

// newLazyFun records the symbol's name and FFI signature. Resolution
// is deferred until the first Call.
func newLazyFun(name string, ret *ffi.Type, args ...*ffi.Type) *lazyFun {
	return &lazyFun{name: name, ret: ret, args: args}
}

// Call resolves the symbol against libHandle on first invocation, then
// forwards to ffi.Fun.Call. If the symbol is absent in the loaded
// library, every call panics with a clear message including the symbol
// name. ret/args follow the same any-typed contract as ffi.Fun.Call.
func (l *lazyFun) Call(ret any, args ...any) {
	l.once.Do(func() {
		l.fun, l.err = libHandle.Prep(l.name, l.ret, l.args...)
	})
	if l.err != nil {
		panic(fmt.Errorf("litertlm: missing C symbol %q in loaded library "+
			"(refresh the prebuilt LiteRT-LM libs to a build that exports it): %w",
			l.name, l.err))
	}
	l.fun.Call(ret, args...)
}
