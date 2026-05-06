package litertlm

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
)

// defaultSchemaInstruction is a Printf format string with one %s
// placeholder for the shape hint. Tuned for instruction-following
// local LLMs — short, imperative, calls out the no-fences rule
// because models love to wrap output in ```json ... ```.
const defaultSchemaInstruction = "Respond with valid JSON only — no commentary, no markdown fences.\n" +
	"The output must match this shape:\n%s"

// GenerateData is the text-only convenience wrapper over
// GenerateDataMulti. It augments prompt with a JSON-shape instruction
// derived from T, generates, extracts the JSON from the (possibly
// markdown-fenced, prose-padded) reply, and unmarshals into *T.
//
// Tier A (this implementation): the augmentation is purely
// prompt-engineered; nothing prevents the model from emitting invalid
// JSON. Tier B (constrained decoding) is gated on upstream LiteRT-LM
// exposing a schema-delivery hook.
//
// Use WithRetries(n) to retry on parse failures (default 0 — one
// attempt). Generate-phase errors (ctx cancellation, FFI failure) are
// returned immediately and do NOT trigger retries.
//
// On parse failure after the final attempt the caller receives a
// *GenerateDataError; use errors.As to inspect Phase / Raw / Attempts.
func GenerateData[T any](ctx context.Context, c *Client, prompt string, opts ...GenOption) (*T, error) {
	return GenerateDataMulti[T](ctx, c, []Part{Text(prompt)}, opts...)
}

// GenerateDataMulti is the multimodal sibling of GenerateData. parts
// may carry text, image, and audio segments; the schema instruction
// is prepended to the LAST text Part (or appended as a new Text part
// when none exists).
//
// All other semantics — retries, GenerateDataError, Tier A vs B —
// match GenerateData.
func GenerateDataMulti[T any](ctx context.Context, c *Client, parts []Part, opts ...GenOption) (*T, error) {
	if c == nil {
		return nil, fmt.Errorf("litertlm: GenerateDataMulti: nil client")
	}

	cfg := resolveGenConfig(opts)

	t := reflect.TypeFor[T]()
	shape, err := shapeOf(t)
	if err != nil {
		return nil, fmt.Errorf("litertlm: GenerateDataMulti: %w", err)
	}

	instruction := cfg.schemaInstruction
	if instruction == "" {
		instruction = defaultSchemaInstruction
	}
	augmentedParts := injectSchema(parts, fmt.Sprintf(instruction, shape))
	wantArray := isArrayType(t)

	var lastErr error
	for attempt := 1; attempt <= 1+cfg.retries; attempt++ {
		if cerr := ctx.Err(); cerr != nil {
			return nil, cerr
		}

		// Hand the already-resolved cfg straight to the internal
		// generateMulti helper. Going through Client.GenerateMulti
		// would re-apply the opts to a fresh genConfig — skip the
		// round trip.
		text, err := c.generateMulti(ctx, augmentedParts, cfg)
		if err != nil {
			return nil, &GenerateDataError{
				Phase:    "generate",
				Err:      err,
				Attempts: attempt,
			}
		}

		extracted, err := extractJSON(text, wantArray)
		if err != nil {
			lastErr = &GenerateDataError{
				Phase:    "parse",
				Err:      err,
				Raw:      text,
				Attempts: attempt,
			}
			continue
		}

		out := new(T)
		if err := json.Unmarshal([]byte(extracted), out); err != nil {
			lastErr = &GenerateDataError{
				Phase:    "parse",
				Err:      err,
				Raw:      text,
				Attempts: attempt,
			}
			continue
		}
		return out, nil
	}
	return nil, lastErr
}

// GenerateDataError describes a structured-output failure. Phase
// distinguishes "generate" (the underlying model call failed) from
// "parse" (the model produced text but it could not be unmarshalled
// into T). Raw holds the model output for parse-phase failures so
// callers can log or display the offending response.
//
// Use errors.As to extract this from a GenerateData error return:
//
//	var gd *GenerateDataError
//	if errors.As(err, &gd) && gd.Phase == "parse" { ... }
type GenerateDataError struct {
	Phase    string // "generate" or "parse"
	Err      error
	Raw      string // populated on parse-phase failures
	Attempts int    // 1-indexed
}

func (e *GenerateDataError) Error() string {
	if e == nil {
		return "<nil GenerateDataError>"
	}
	return fmt.Sprintf("litertlm: GenerateData %s phase failed after %d attempt(s): %v",
		e.Phase, e.Attempts, e.Err)
}

func (e *GenerateDataError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// injectSchema returns a copy of parts with instruction prepended to
// the LAST text Part — matching GenerateData[T]'s "instruction, then
// prompt" ordering. When no text part exists, a new Text(instruction)
// is appended at the end (works for image/audio-only inputs).
//
// The caller's slice is never mutated.
func injectSchema(parts []Part, instruction string) []Part {
	out := append([]Part(nil), parts...)
	for i := len(out) - 1; i >= 0; i-- {
		if out[i].kind == partText {
			out[i].text = instruction + "\n\n" + out[i].text
			return out
		}
	}
	return append(out, Text(instruction))
}
