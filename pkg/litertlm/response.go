package litertlm

// Response wraps a generation result, exposing per-candidate text
// plus optional score and token-length accessors. Candidates are
// copied out of the engine when the generation completes; a Response
// holds no native resources and needs no explicit release.
type Response struct {
	candidates []Candidate
	// bench is populated when WithBenchmarkEnabled was set at New
	// time; nil otherwise.
	bench *Benchmark
}

// newResponse wraps the candidates of one completed generation.
func newResponse(candidates []Candidate) *Response {
	return &Response{candidates: candidates}
}

// newTextResponse constructs a Response carrying a single text
// candidate. Score and TokenLength return (0, false) on the
// resulting Response.
func newTextResponse(s string) *Response {
	if s == "" {
		return &Response{}
	}
	return &Response{candidates: []Candidate{{Text: s}}}
}

// Text returns the first candidate's text. Returns "" when no
// candidates were produced.
func (r *Response) Text() string {
	return r.Candidate(0)
}

// NumCandidates returns the number of candidates the engine emitted.
func (r *Response) NumCandidates() int {
	if r == nil {
		return 0
	}
	return len(r.candidates)
}

// Candidate returns the text of the i-th candidate. Returns "" for
// out-of-range indices.
func (r *Response) Candidate(i int) string {
	if r == nil || i < 0 || i >= len(r.candidates) {
		return ""
	}
	return r.candidates[i].Text
}

// Score returns the candidate's score at index i. ok mirrors the
// engine's has-score predicate, which fires for any in-range index —
// see Responses.Score for the (0, true) placeholder caveat applicable
// to non-scoring sources (Generate / GenerateStream return placeholder
// values; only ScoreTexts populates real ones).
func (r *Response) Score(i int) (float32, bool) {
	if r == nil || i < 0 || i >= len(r.candidates) || r.candidates[i].Score == nil {
		return 0, false
	}
	return *r.candidates[i].Score, true
}

// TokenLength returns the candidate's tokenized length at index i.
// Populated only when the producer attached lengths (currently just
// ScoreTexts with storeTokenLengths=true).
func (r *Response) TokenLength(i int) (int, bool) {
	if r == nil || i < 0 || i >= len(r.candidates) || r.candidates[i].TokenLength == nil {
		return 0, false
	}
	return *r.candidates[i].TokenLength, true
}

// Benchmark returns the captured per-generation benchmark, or nil if
// the Client wasn't constructed with WithBenchmarkEnabled.
func (r *Response) Benchmark() *Benchmark {
	if r == nil {
		return nil
	}
	return r.bench
}

// TokenScores returns the per-token scores for candidate i. ok is
// false when the producing source did not attach per-token scores.
func (r *Response) TokenScores(i int) ([]float32, bool) {
	if r == nil || i < 0 || i >= len(r.candidates) || r.candidates[i].TokenScores == nil {
		return nil, false
	}
	return r.candidates[i].TokenScores, true
}
