package litertlm

// Response wraps the result of Client.GenerateResponse /
// Client.GenerateMultiResponse, exposing candidate text and optional
// benchmark metrics captured on that call.
type Response struct {
	text  string
	bench *Benchmark
}

// newTextResponse constructs a Response carrying a single text candidate.
func newTextResponse(s string) *Response {
	return &Response{text: s}
}

// Text returns the first candidate's text, or "" on nil/empty Response.
func (r *Response) Text() string {
	if r == nil {
		return ""
	}
	return r.text
}

// NumCandidates returns 1 when the response contains text and 0 otherwise.
func (r *Response) NumCandidates() int {
	if r == nil || r.text == "" {
		return 0
	}
	return 1
}

// Candidate returns the text of the i-th candidate, or "" when out of range.
func (r *Response) Candidate(i int) string {
	if r == nil || i != 0 {
		return ""
	}
	return r.text
}

// Benchmark returns the captured per-generation benchmark, or nil if
// the Client wasn't constructed with WithBenchmarkEnabled.
func (r *Response) Benchmark() *Benchmark {
	if r == nil {
		return nil
	}
	return r.bench
}
