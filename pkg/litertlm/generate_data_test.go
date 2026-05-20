package litertlm

import (
	"context"
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

// TestInjectSchema_PrependsToOnlyTextPart: single text part is
// rewritten as "instruction\n\nprompt".
func TestInjectSchema_PrependsToOnlyTextPart(t *testing.T) {
	parts := []Part{Text("describe the image")}
	out := injectSchema(parts, "INSTR")
	if len(out) != 1 {
		t.Fatalf("len = %d, want 1", len(out))
	}
	want := "INSTR\n\ndescribe the image"
	if out[0].text != want {
		t.Errorf("text = %q, want %q", out[0].text, want)
	}
}

// TestInjectSchema_PicksLastTextPart: when multiple text parts are
// present, the LAST one is rewritten so the schema instruction lands
// next to the most recent user prompt.
func TestInjectSchema_PicksLastTextPart(t *testing.T) {
	parts := []Part{
		Text("system context"),
		Image([]byte{0xFF}),
		Text("the user prompt"),
	}
	out := injectSchema(parts, "INSTR")
	if len(out) != 3 {
		t.Fatalf("len = %d, want 3", len(out))
	}
	if out[0].text != "system context" {
		t.Errorf("first text mutated: %q", out[0].text)
	}
	if !strings.HasPrefix(out[2].text, "INSTR\n\n") {
		t.Errorf("last text not augmented: %q", out[2].text)
	}
}

// TestInjectSchema_AppendsWhenNoText: image-only / audio-only inputs
// get a synthesized text Part with just the instruction.
func TestInjectSchema_AppendsWhenNoText(t *testing.T) {
	parts := []Part{Image([]byte{0xFF}), Audio([]byte("riff"))}
	out := injectSchema(parts, "INSTR")
	if len(out) != 3 {
		t.Fatalf("len = %d, want 3", len(out))
	}
	if !out[2].IsText() || out[2].text != "INSTR" {
		t.Errorf("appended part = %+v, want Text(INSTR)", out[2])
	}
}

// TestInjectSchema_DoesNotMutateInput: caller's slice is preserved.
func TestInjectSchema_DoesNotMutateInput(t *testing.T) {
	parts := []Part{Text("original")}
	_ = injectSchema(parts, "INSTR")
	if parts[0].text != "original" {
		t.Errorf("input mutated: %q", parts[0].text)
	}
}

// ---- augmentForToolUse --------------------------------------------------

func TestAugmentForToolUse_PrependsToOnlyTextPart(t *testing.T) {
	parts := []Part{Text("extract entities")}
	out := augmentForToolUse(parts)
	if len(out) != 1 {
		t.Fatalf("len = %d, want 1", len(out))
	}
	if !strings.HasPrefix(out[0].text, captureDirective) {
		t.Errorf("missing directive prefix: %q", out[0].text)
	}
	if !strings.HasSuffix(out[0].text, "extract entities") {
		t.Errorf("original prompt missing from output: %q", out[0].text)
	}
}

func TestAugmentForToolUse_PicksLastTextPart(t *testing.T) {
	parts := []Part{
		Text("system context"),
		Image([]byte{0xFF}),
		Text("the user prompt"),
	}
	out := augmentForToolUse(parts)
	if len(out) != 3 {
		t.Fatalf("len = %d, want 3", len(out))
	}
	if out[0].text != "system context" {
		t.Errorf("first text mutated: %q", out[0].text)
	}
	if !strings.HasPrefix(out[2].text, captureDirective) {
		t.Errorf("last text not augmented: %q", out[2].text)
	}
}

func TestAugmentForToolUse_AppendsWhenNoText(t *testing.T) {
	parts := []Part{Image([]byte{0xFF}), Audio([]byte("riff"))}
	out := augmentForToolUse(parts)
	if len(out) != 3 {
		t.Fatalf("len = %d, want 3", len(out))
	}
	if !out[2].IsText() || out[2].text != captureDirective {
		t.Errorf("appended part = %+v, want Text(captureDirective)", out[2])
	}
}

func TestAugmentForToolUse_DoesNotMutateInput(t *testing.T) {
	parts := []Part{Text("original")}
	_ = augmentForToolUse(parts)
	if parts[0].text != "original" {
		t.Errorf("input mutated: %q", parts[0].text)
	}
}

// ---- tryCaptureToolSilent ------------------------------------------------

// Unsuitable T (slice) returns nil immediately without touching the
// engine. Verifies the silent-fallback contract: no panic, no error
// surface, no chat construction attempt.
func TestTryCaptureToolSilent_UnsuitableType(t *testing.T) {
	c := &Client{}
	if got := tryCaptureToolSilent[[]capturePerson](context.Background(), c, []Part{Text("x")}); got != nil {
		t.Errorf("unsuitable T returned non-nil: %+v", got)
	}
	if len(c.tools) != 0 {
		t.Errorf("unsuitable T leaked a tool into the registry: %v", c.tools)
	}
}

// Nil-client safety: returns nil without panic. capturePerson is
// suitable, so without this guard the function would crash on the
// first c.tools access inside getOrSynthesizeCaptureTool.
func TestTryCaptureToolSilent_NilClient(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("tryCaptureToolSilent panicked on nil client: %v", r)
		}
	}()
	if got := tryCaptureToolSilent[capturePerson](context.Background(), nil, []Part{Text("x")}); got != nil {
		t.Errorf("nil client returned non-nil: %+v", got)
	}
}
