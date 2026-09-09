package litertlm

import "testing"

func TestResponse_Accessors(t *testing.T) {
	var nilResp *Response
	checkEmptyResponse(t, nilResp, "nil")

	empty := &Response{}
	checkEmptyResponse(t, empty, "empty")

	r := newTextResponse("hello")
	if r.Text() != "hello" {
		t.Errorf("Text = %q, want hello", r.Text())
	}
	if r.NumCandidates() != 1 {
		t.Errorf("NumCandidates = %d, want 1", r.NumCandidates())
	}
	if r.Candidate(0) != "hello" {
		t.Errorf("Candidate(0) = %q, want hello", r.Candidate(0))
	}
	if r.Candidate(1) != "" {
		t.Errorf("Candidate(1) = %q, want empty", r.Candidate(1))
	}
}

func checkEmptyResponse(t *testing.T, r *Response, label string) {
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
}
