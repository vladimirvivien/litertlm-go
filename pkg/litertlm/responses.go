package litertlm

import "unsafe"

// Delete releases a Responses handle.
func (r Responses) Delete() { deleteHandle(r, responsesDeleteFunc) }

// NumCandidates returns the number of response candidates available.
func (r Responses) NumCandidates() int {
	return int(getHandleScalar[Responses, uint64](r, responsesGetNumCandidatesFunc, "responses_get_num_candidates", 0))
}

// Text returns the response text at index i, copied into Go memory.
// Returns "" if the index is out of bounds or the underlying pointer is
// null. The copy lets the result outlive Responses.Delete().
func (r Responses) Text(i int) string {
	if r == 0 || i < 0 {
		return ""
	}
	idx := uint64(i)
	s, ok, err := callStatusString(responsesGetResponseTextAtFunc, "responses_get_response_text_at",
		unsafe.Pointer(&r), unsafe.Pointer(&idx))
	if err != nil || !ok {
		return ""
	}
	return s
}

// Score returns the candidate's score at index i. ok mirrors the C
// `has_score_at` predicate, which returns true whenever a score slot
// exists for the index — *not* whether a meaningful score was computed.
// Only ScoreTexts populates real values; GenerateContent / RunDecode
// candidates report (0, true) as a placeholder.
func (r Responses) Score(i int) (float32, bool) {
	if r == 0 || i < 0 {
		return 0, false
	}
	idx := uint64(i)
	has, err := callStatusBool(responsesHasScoreAtFunc, "responses_has_score_at",
		unsafe.Pointer(&r), unsafe.Pointer(&idx))
	if err != nil || !has {
		return 0, false
	}
	v, err := callStatusScalar[float32](responsesGetScoreAtFunc, "responses_get_score_at",
		unsafe.Pointer(&r), unsafe.Pointer(&idx))
	if err != nil {
		return 0, false
	}
	return v, true
}

// TokenLength returns the candidate's tokenized length at index i. ok
// is false when no length was stored — most often when ScoreTexts was
// called with storeTokenLengths=false.
func (r Responses) TokenLength(i int) (int, bool) {
	if r == 0 || i < 0 {
		return 0, false
	}
	idx := uint64(i)
	has, err := callStatusBool(responsesHasTokenLengthAtFunc, "responses_has_token_length_at",
		unsafe.Pointer(&r), unsafe.Pointer(&idx))
	if err != nil || !has {
		return 0, false
	}
	n, err := callStatusScalar[uint64](responsesGetTokenLengthAtFunc, "responses_get_token_length_at",
		unsafe.Pointer(&r), unsafe.Pointer(&idx))
	if err != nil {
		return 0, false
	}
	return int(n), true
}

// TokenScores returns the per-token scores for candidate i, or
// (nil, false) when no per-token scores are present at that index.
// The returned slice is copied into Go memory and is safe to retain
// after the underlying Responses handle is deleted.
func (r Responses) TokenScores(i int) ([]float32, bool) {
	if r == 0 || i < 0 {
		return nil, false
	}
	idx := uint64(i)
	has, err := callStatusBool(responsesHasTokenScoresAtFunc, "responses_has_token_scores_at",
		unsafe.Pointer(&r), unsafe.Pointer(&idx))
	if err != nil || !has {
		return nil, false
	}
	n, err := callStatusScalar[uint64](responsesGetNumTokenScoresAtFunc, "responses_get_num_token_scores_at",
		unsafe.Pointer(&r), unsafe.Pointer(&idx))
	if err != nil {
		return nil, false
	}
	if n == 0 {
		return []float32{}, true
	}
	ptr, err := callStatusScalar[*float32](responsesGetTokenScoresAtFunc, "responses_get_token_scores_at",
		unsafe.Pointer(&r), unsafe.Pointer(&idx))
	if err != nil || ptr == nil {
		return nil, false
	}
	return copyCSlice(ptr, n), true
}
