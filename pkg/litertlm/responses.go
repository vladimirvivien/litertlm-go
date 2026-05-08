package litertlm

import (
	"unsafe"

	"github.com/vladimirvivien/litertlm-go/pkg/utils"
)

// Delete releases a Responses handle.
func (r Responses) Delete() {
	if r == 0 {
		return
	}
	responsesDeleteFunc.Call(nil, unsafe.Pointer(&r))
}

// NumCandidates returns the number of response candidates available.
func (r Responses) NumCandidates() int {
	if r == 0 {
		return 0
	}
	var n int32
	responsesGetNumCandidatesFunc.Call(unsafe.Pointer(&n), unsafe.Pointer(&r))
	return int(n)
}

// Text returns the response text at index i, copied into Go memory.
// Returns "" if the index is out of bounds or the underlying pointer is
// null. The copy lets the result outlive Responses.Delete().
func (r Responses) Text(i int) string {
	if r == 0 {
		return ""
	}
	var ptr *byte
	responsesGetResponseTextAtFunc.Call(
		unsafe.Pointer(&ptr),
		unsafe.Pointer(&r),
		unsafe.Pointer(new(int32(i))),
	)
	if ptr == nil {
		return ""
	}
	return utils.BytePtrToString(ptr)
}

// Score returns the candidate's score at index i. ok mirrors the C
// `has_score_at` predicate, which returns true whenever a score slot
// exists for the index — *not* whether a meaningful score was computed.
// Only ScoreTexts populates real values; GenerateContent / RunDecode
// candidates report (0, true) as a placeholder. Use the producing
// method to decide whether a returned score is meaningful.
func (r Responses) Score(i int) (float32, bool) {
	if r == 0 {
		return 0, false
	}
	var has uint8
	responsesHasScoreAtFunc.Call(
		unsafe.Pointer(&has),
		unsafe.Pointer(&r),
		unsafe.Pointer(new(int32(i))),
	)
	if has == 0 {
		return 0, false
	}
	var v float32
	responsesGetScoreAtFunc.Call(
		unsafe.Pointer(&v),
		unsafe.Pointer(&r),
		unsafe.Pointer(new(int32(i))),
	)
	return v, true
}

// TokenLength returns the candidate's tokenized length at index i. ok
// is false when no length was stored — most often when ScoreTexts was
// called with storeTokenLengths=false.
func (r Responses) TokenLength(i int) (int, bool) {
	if r == 0 {
		return 0, false
	}
	var has uint8
	responsesHasTokenLengthAtFunc.Call(
		unsafe.Pointer(&has),
		unsafe.Pointer(&r),
		unsafe.Pointer(new(int32(i))),
	)
	if has == 0 {
		return 0, false
	}
	var v int32
	responsesGetTokenLengthAtFunc.Call(
		unsafe.Pointer(&v),
		unsafe.Pointer(&r),
		unsafe.Pointer(new(int32(i))),
	)
	return int(v), true
}

// TokenScores returns the per-token scores for candidate i, or
// (nil, false) when no per-token scores are present at that index.
// The returned slice is copied into Go memory and is safe to retain
// after the underlying Responses handle is deleted.
func (r Responses) TokenScores(i int) ([]float32, bool) {
	if r == 0 {
		return nil, false
	}
	var has uint8
	responsesHasTokenScoresAtFunc.Call(
		unsafe.Pointer(&has),
		unsafe.Pointer(&r),
		unsafe.Pointer(new(int32(i))),
	)
	if has == 0 {
		return nil, false
	}
	var n int32
	responsesGetNumTokenScoresAtFunc.Call(
		unsafe.Pointer(&n),
		unsafe.Pointer(&r),
		unsafe.Pointer(new(int32(i))),
	)
	if n <= 0 {
		return []float32{}, true
	}
	var ptr *float32
	responsesGetTokenScoresAtFunc.Call(
		unsafe.Pointer(&ptr),
		unsafe.Pointer(&r),
		unsafe.Pointer(new(int32(i))),
	)
	if ptr == nil {
		return nil, false
	}
	out := make([]float32, n)
	src := unsafe.Slice(ptr, n)
	copy(out, src)
	return out, true
}
