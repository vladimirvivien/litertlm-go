package litertlm

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/vladimirvivien/litertlm-go/internal/cstr"
)

// tokenizeResult and detokenizeResult wrap the C API's opaque
// LiteRtLmTokenizeResult* / LiteRtLmDetokenizeResult* handles.
type (
	tokenizeResult   uintptr
	detokenizeResult uintptr
)

// Tokenize splits text into the engine tokenizer's id sequence.
// The returned slice is owned by the caller and survives any
// subsequent engine operations.
func (e Engine) Tokenize(text string) ([]int32, error) {
	if e == 0 {
		return nil, fmt.Errorf("litertlm: engine_tokenize: invalid engine")
	}
	textPtr, err := cstr.BytePtrFromString(text)
	if err != nil {
		return nil, err
	}

	r, err := callStatusHandle[tokenizeResult](engineTokenizeFunc, "engine_tokenize",
		unsafe.Pointer(&e),
		unsafe.Pointer(&textPtr),
	)
	runtime.KeepAlive(textPtr)
	if err != nil {
		return nil, err
	}
	defer r.delete()

	n, err := callStatusScalar[uint64](tokenizeResultGetNumTokensFunc, "tokenize_result_get_num_tokens",
		unsafe.Pointer(&r))
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return []int32{}, nil
	}

	ptr, err := callStatusScalar[*int32](tokenizeResultGetTokensFunc, "tokenize_result_get_tokens",
		unsafe.Pointer(&r))
	if err != nil {
		return nil, err
	}
	return copyCSlice(ptr, n), nil
}

// Detokenize reverses Tokenize, decoding ids into a UTF-8 string.
// LiteRT-LM passes through SentencePiece's internal space marker
// (U+2581 ▁) without normalising.
func (e Engine) Detokenize(tokens []int32) (string, error) {
	if e == 0 {
		return "", fmt.Errorf("litertlm: engine_detokenize: invalid engine")
	}
	if len(tokens) == 0 {
		return "", nil
	}

	tokensPtr := unsafe.Pointer(&tokens[0])
	num := uint64(len(tokens))

	r, err := callStatusHandle[detokenizeResult](engineDetokenizeFunc, "engine_detokenize",
		unsafe.Pointer(&e),
		unsafe.Pointer(&tokensPtr),
		unsafe.Pointer(&num),
	)
	runtime.KeepAlive(tokens)
	if err != nil {
		return "", err
	}
	defer r.delete()

	s, ok, err := callStatusString(detokenizeResultGetStringFunc, "detokenize_result_get_string",
		unsafe.Pointer(&r))
	if err != nil {
		return "", err
	}
	if !ok {
		return "", nil
	}
	return s, nil
}

func (r tokenizeResult) delete()   { deleteHandle(r, tokenizeResultDeleteFunc) }
func (r detokenizeResult) delete() { deleteHandle(r, detokenizeResultDeleteFunc) }

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
// configured.
func (e Engine) StartTokenIDs() ([]int32, error) {
	if e == 0 {
		return nil, fmt.Errorf("litertlm: engine_get_start_token: invalid engine")
	}
	tu, err := callStatusHandle[tokenUnion](engineGetStartTokenFunc, "engine_get_start_token",
		unsafe.Pointer(&e))
	if err != nil {
		return nil, err
	}
	if tu == 0 {
		return nil, nil
	}
	defer tu.delete()
	return e.tokenUnionToIDs(tu)
}

// StopTokenIDs returns the model's configured stop tokens (EOS) as a
// slice of id sequences — one inner slice per stop sequence.
func (e Engine) StopTokenIDs() ([][]int32, error) {
	if e == 0 {
		return nil, fmt.Errorf("litertlm: engine_get_stop_tokens: invalid engine")
	}
	tus, err := callStatusHandle[tokenUnions](engineGetStopTokensFunc, "engine_get_stop_tokens",
		unsafe.Pointer(&e))
	if err != nil {
		return nil, err
	}
	if tus == 0 {
		return nil, nil
	}
	defer tus.delete()

	n, err := callStatusScalar[uint64](tokenUnionsGetNumTokensFunc, "token_unions_get_num_tokens",
		unsafe.Pointer(&tus))
	if err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, nil
	}

	out := make([][]int32, 0, n)
	for i := range n {
		idx := i
		inner, innerErr := callStatusHandle[tokenUnion](tokenUnionsGetTokenAtFunc, "token_unions_get_token_at",
			unsafe.Pointer(&tus), unsafe.Pointer(&idx))
		if innerErr != nil {
			return nil, innerErr
		}
		if inner == 0 {
			continue
		}
		// Inner unions are borrowed views into tus's storage; tus.delete() frees all.
		ids, idsErr := e.tokenUnionToIDs(inner)
		if idsErr != nil {
			return nil, idsErr
		}
		out = append(out, ids)
	}
	return out, nil
}

// tokenUnionToIDs reads the union's discriminator and dispatches to
// either a direct id-array copy or Tokenize for the string form.
func (e Engine) tokenUnionToIDs(tu tokenUnion) ([]int32, error) {
	typ, err := callStatusScalar[int32](tokenUnionGetTypeFunc, "token_union_get_type",
		unsafe.Pointer(&tu))
	if err != nil {
		return nil, err
	}

	switch typ {
	case tokenUnionTypeIDs:
		var ptr *int32
		var n uint64
		ptrAddr := &ptr
		nAddr := &n
		err = callStatus(tokenUnionGetIdsFunc, "token_union_get_ids",
			unsafe.Pointer(&tu),
			unsafe.Pointer(&ptrAddr),
			unsafe.Pointer(&nAddr),
		)
		runtime.KeepAlive(ptrAddr)
		runtime.KeepAlive(nAddr)
		if err != nil {
			return nil, err
		}
		return copyCSlice(ptr, n), nil

	case tokenUnionTypeString:
		s, ok, strErr := callStatusString(tokenUnionGetStringFunc, "token_union_get_string",
			unsafe.Pointer(&tu))
		if strErr != nil {
			return nil, strErr
		}
		if !ok {
			return []int32{}, nil
		}
		return e.Tokenize(s)

	default:
		return nil, fmt.Errorf("litertlm: unknown token_union type %d", typ)
	}
}

func (tu tokenUnion) delete()   { deleteHandle(tu, tokenUnionDeleteFunc) }
func (tus tokenUnions) delete() { deleteHandle(tus, tokenUnionsDeleteFunc) }
