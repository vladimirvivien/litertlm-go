package litertlm

import "runtime"

// Response wraps a generation result, exposing per-candidate text
// plus optional score and token-length accessors.
//
// Lifetime: when the *Response becomes unreachable, a
// runtime.AddCleanup-registered callback frees any underlying C
// handle automatically. Callers do not (and should not) call Delete
// on a Response.
//
// runtime.AddCleanup is best-effort — the cleanup may run a few GC
// cycles later than the variable goes out of scope. If deterministic
// release matters (tight loop generating many responses,
// memory-bound), drop down to the low-level Session API and call
// .Delete() yourself.
type Response struct {
	handle Responses
	// text holds a synthesized single-candidate reply when no C
	// handle is available; Score / TokenLength return (0, false) in
	// that case.
	text string
}

// newResponse wraps a freshly-created Responses handle and registers
// the cleanup. The cleanup captures `h` by value (a uintptr), not the
// *Response itself — that's what AddCleanup demands so the cleanup
// arg can't keep the wrapper reachable.
func newResponse(h Responses) *Response {
	r := &Response{handle: h}
	runtime.AddCleanup(r, func(handle Responses) {
		handle.Delete()
	}, h)
	return r
}

// newTextResponse constructs a Response carrying a single text
// candidate without a backing C handle. Score and TokenLength
// return (0, false) on the resulting Response.
func newTextResponse(s string) *Response {
	return &Response{text: s}
}

// Text returns the first candidate's text. Returns "" when the
// underlying handle is null or no candidates were produced.
func (r *Response) Text() string {
	if r == nil {
		return ""
	}
	if r.handle == 0 {
		return r.text
	}
	return r.handle.Text(0)
}

// NumCandidates returns the number of candidates the engine emitted.
func (r *Response) NumCandidates() int {
	if r == nil {
		return 0
	}
	if r.handle == 0 {
		if r.text != "" {
			return 1
		}
		return 0
	}
	return r.handle.NumCandidates()
}

// Candidate returns the text of the i-th candidate. Returns "" for
// out-of-range indices.
func (r *Response) Candidate(i int) string {
	if r == nil {
		return ""
	}
	if r.handle == 0 {
		if i == 0 {
			return r.text
		}
		return ""
	}
	return r.handle.Text(i)
}

// Score returns the candidate's score at index i. ok mirrors the C
// has_score_at predicate, which fires for any in-range index — see
// Responses.Score for the (0, true) placeholder caveat applicable to
// non-scoring sources (Generate / GenerateStream return placeholder
// values; only ScoreTexts populates real ones).
func (r *Response) Score(i int) (float32, bool) {
	if r == nil || r.handle == 0 {
		return 0, false
	}
	return r.handle.Score(i)
}

// TokenLength returns the candidate's tokenized length at index i.
// Populated only when the producer attached lengths (currently just
// ScoreTexts with storeTokenLengths=true).
func (r *Response) TokenLength(i int) (int, bool) {
	if r == nil || r.handle == 0 {
		return 0, false
	}
	return r.handle.TokenLength(i)
}
