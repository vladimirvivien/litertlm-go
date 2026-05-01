package litertlm

import (
	"errors"
	"strings"
	"testing"
)

func TestGenerateDataError_Format(t *testing.T) {
	inner := errors.New("invalid character 'x'")
	e := &GenerateDataError{
		Phase:    "parse",
		Err:      inner,
		Raw:      "xyz",
		Attempts: 3,
	}
	got := e.Error()
	for _, want := range []string{"parse", "3 attempt", "invalid character"} {
		if !strings.Contains(got, want) {
			t.Errorf("Error() = %q, want substring %q", got, want)
		}
	}
}

func TestGenerateDataError_Unwrap(t *testing.T) {
	inner := errors.New("inner")
	e := &GenerateDataError{Phase: "parse", Err: inner}
	if !errors.Is(e, inner) {
		t.Errorf("errors.Is should reach inner via Unwrap")
	}

	// Targeted As-style extraction.
	wrapped := &GenerateDataError{
		Phase: "generate",
		Err:   inner,
	}
	var gd *GenerateDataError
	if !errors.As(wrapped, &gd) {
		t.Fatalf("errors.As should match *GenerateDataError")
	}
	if gd.Phase != "generate" {
		t.Errorf("As'd error phase = %q, want generate", gd.Phase)
	}
}

func TestGenerateDataError_NilSafe(t *testing.T) {
	var e *GenerateDataError
	if msg := e.Error(); msg == "" {
		t.Errorf("nil *GenerateDataError should still produce a sentinel string")
	}
	if got := e.Unwrap(); got != nil {
		t.Errorf("nil.Unwrap() = %v, want nil", got)
	}
}
