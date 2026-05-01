package litertlm

import "testing"

// TestResponse_NilSafe covers the defensive paths in every accessor:
// callers should never see a panic for a nil *Response or one whose
// handle is zero. Pure-Go (no FFI required).
func TestResponse_NilSafe(t *testing.T) {
	var r *Response
	checkZero(t, r, "nil")

	r = &Response{} // explicit zero handle
	checkZero(t, r, "zero handle")
}

func checkZero(t *testing.T, r *Response, label string) {
	t.Helper()
	if r.Text() != "" {
		t.Errorf("%s: Text = %q, want empty", label, r.Text())
	}
	if r.NumCandidates() != 0 {
		t.Errorf("%s: NumCandidates = %d, want 0", label, r.NumCandidates())
	}
	if r.Candidate(0) != "" {
		t.Errorf("%s: Candidate(0) = %q, want empty", label, r.Candidate(0))
	}
	if v, ok := r.Score(0); v != 0 || ok {
		t.Errorf("%s: Score(0) = (%v, %v), want (0, false)", label, v, ok)
	}
	if v, ok := r.TokenLength(0); v != 0 || ok {
		t.Errorf("%s: TokenLength(0) = (%v, %v), want (0, false)", label, v, ok)
	}
}
