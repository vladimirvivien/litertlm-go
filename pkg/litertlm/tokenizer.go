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

// tokenUnion / tokenUnions wrap the C handles for start/stop token
// metadata. Both stay unexported — callers see []int32 / [][]int32
// via Engine.StartTokenIDs / Engine.StopTokenIDs.
type (
	tokenUnion  uintptr
	tokenUnions uintptr
)

// LiteRtLmTokenUnionType values (see c/engine.h).
const (
	tokenUnionTypeString int32 = 0
	tokenUnionTypeIDs    int32 = 1
)

// StartTokenIDs returns the model's configured start token (BOS) as a
// slice of token ids. Returns (nil, nil) when no start token is
// configured. String-form unions are run through Tokenize so callers
// always see ids.
func (e Engine) StartTokenIDs() ([]int32, error) {
	if e == 0 {
		return nil, fmt.Errorf("litertlm: engine_get_start_token: invalid engine")
	}
	var tu tokenUnion
	engineGetStartTokenFunc.Call(unsafe.Pointer(&tu), unsafe.Pointer(&e))
	if tu == 0 {
		return nil, nil
	}
	defer tu.delete()
	return e.tokenUnionToIDs(tu)
}

// StopTokenIDs returns the model's configured stop tokens (EOS) as a
// slice of id sequences — one inner slice per stop sequence. Returns
// (nil, nil) when no stop tokens are configured. String-form sequences
// are run through Tokenize.
func (e Engine) StopTokenIDs() ([][]int32, error) {
	if e == 0 {
		return nil, fmt.Errorf("litertlm: engine_get_stop_tokens: invalid engine")
	}
	var tus tokenUnions
	engineGetStopTokensFunc.Call(unsafe.Pointer(&tus), unsafe.Pointer(&e))
	if tus == 0 {
		return nil, nil
	}
	defer tus.delete()

	var n uint64
	tokenUnionsGetNumTokensFunc.Call(unsafe.Pointer(&n), unsafe.Pointer(&tus))
	if n == 0 {
		return nil, nil
	}

	out := make([][]int32, 0, n)
	for i := uint64(0); i < n; i++ {
		var inner tokenUnion
		tokenUnionsGetTokenAtFunc.Call(
			unsafe.Pointer(&inner),
			unsafe.Pointer(&tus),
			unsafe.Pointer(new(i)),
		)
		if inner == 0 {
			continue
		}
		// Inner unions are borrowed views into the parent's storage
		// (see c/engine.cc litert_lm_token_unions_get_token_at) — do
		// NOT call delete() on them; tus.delete() frees the lot.
		ids, err := e.tokenUnionToIDs(inner)
		if err != nil {
			return nil, err
		}
		out = append(out, ids)
	}
	return out, nil
}

// tokenUnionToIDs reads the union's discriminator and dispatches to
// either a direct id-array copy or Tokenize for the string form.
func (e Engine) tokenUnionToIDs(tu tokenUnion) ([]int32, error) {
	var typ int32
	tokenUnionGetTypeFunc.Call(unsafe.Pointer(&typ), unsafe.Pointer(&tu))

	switch typ {
	case tokenUnionTypeIDs:
		// out_tokens (const int**) and out_num_tokens (size_t*) are
		// pointer-to-where-C-should-write. The purego FFI layer reads
		// one level of indirection from the Go-side address, so we
		// need a `**int32` / `*uint64` storage and pass the *address*
		// of those — that's what makes C see `&ptr` rather than the
		// (initially-nil) value of ptr itself.
		var ptr *int32
		var n uint64
		ptrAddr := &ptr
		nAddr := &n
		var status int32
		tokenUnionGetIdsFunc.Call(
			unsafe.Pointer(&status),
			unsafe.Pointer(&tu),
			unsafe.Pointer(&ptrAddr),
			unsafe.Pointer(&nAddr),
		)
		if status != 0 || ptr == nil || n == 0 {
			return []int32{}, nil
		}
		// Copy out before the parent union/result is deleted.
		src := unsafe.Slice(ptr, n)
		out := make([]int32, n)
		copy(out, src)
		return out, nil

	case tokenUnionTypeString:
		var sptr *byte
		tokenUnionGetStringFunc.Call(unsafe.Pointer(&sptr), unsafe.Pointer(&tu))
		if sptr == nil {
			return []int32{}, nil
		}
		return e.Tokenize(utils.BytePtrToString(sptr))

	default:
		return nil, fmt.Errorf("litertlm: unknown token_union type %d", typ)
	}
}

func (tu tokenUnion) delete() {
	if tu == 0 {
		return
	}
	tokenUnionDeleteFunc.Call(nil, unsafe.Pointer(&tu))
}

func (tus tokenUnions) delete() {
	if tus == 0 {
		return
	}
	tokenUnionsDeleteFunc.Call(nil, unsafe.Pointer(&tus))
}
