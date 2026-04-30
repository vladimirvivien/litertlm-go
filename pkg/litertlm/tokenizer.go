package litertlm

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/vladimirvivien/litertlm-go/pkg/utils"
)

// tokenizeResult and detokenizeResult wrap the C API's opaque
// LiteRtLmTokenizeResult* / LiteRtLmDetokenizeResult* handles. Both are
// kept unexported — callers see Go-native slices and strings via
// Engine.Tokenize / Engine.Detokenize.
type (
	tokenizeResult   uintptr
	detokenizeResult uintptr
)

// Tokenize splits text into the engine tokenizer's id sequence.
func (e Engine) Tokenize(text string) ([]int32, error) {
	if e == 0 {
		return nil, fmt.Errorf("litertlm: engine_tokenize: invalid engine")
	}
	textPtr, err := utils.BytePtrFromString(text)
	if err != nil {
		return nil, err
	}

	r, err := callForHandle[tokenizeResult](engineTokenizeFunc, "engine_tokenize",
		unsafe.Pointer(&e),
		unsafe.Pointer(&textPtr),
	)
	if err != nil {
		return nil, err
	}
	defer r.delete()

	var n uint64
	tokenizeResultGetNumTokensFunc.Call(unsafe.Pointer(&n), unsafe.Pointer(&r))
	if n == 0 {
		return []int32{}, nil
	}

	var ptr *int32
	tokenizeResultGetTokensFunc.Call(unsafe.Pointer(&ptr), unsafe.Pointer(&r))
	if ptr == nil {
		return []int32{}, nil
	}
	// The C-side pointer is borrowed for the lifetime of the result —
	// copy into Go memory before defer r.delete() runs.
	src := unsafe.Slice(ptr, n)
	out := make([]int32, n)
	copy(out, src)
	return out, nil
}

// Detokenize reverses Tokenize, decoding ids into a UTF-8 string.
func (e Engine) Detokenize(tokens []int32) (string, error) {
	if e == 0 {
		return "", fmt.Errorf("litertlm: engine_detokenize: invalid engine")
	}
	if len(tokens) == 0 {
		return "", nil
	}

	tokensPtr := unsafe.Pointer(&tokens[0])
	num := uint64(len(tokens))

	r, err := callForHandle[detokenizeResult](engineDetokenizeFunc, "engine_detokenize",
		unsafe.Pointer(&e),
		unsafe.Pointer(&tokensPtr),
		unsafe.Pointer(&num),
	)
	// KeepAlive after the FFI call: the C side has already read
	// through tokensPtr by now, but the explicit reference here
	// stops the compiler from dropping tokens before this line.
	runtime.KeepAlive(tokens)
	if err != nil {
		return "", err
	}
	defer r.delete()

	var strPtr *byte
	detokenizeResultGetStringFunc.Call(unsafe.Pointer(&strPtr), unsafe.Pointer(&r))
	if strPtr == nil {
		return "", nil
	}
	// Copy into Go memory so the result outlives defer r.delete().
	return utils.BytePtrToString(strPtr), nil
}

func (r tokenizeResult) delete() {
	if r == 0 {
		return
	}
	tokenizeResultDeleteFunc.Call(nil, unsafe.Pointer(&r))
}

func (r detokenizeResult) delete() {
	if r == 0 {
		return
	}
	detokenizeResultDeleteFunc.Call(nil, unsafe.Pointer(&r))
}
