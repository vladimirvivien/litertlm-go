package litertlm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
)

// GenerateData[T] / GenerateDataMulti[T] route structured output
// through a synthesized capture tool when T is a struct. The model
// invokes that tool; the C-side model_data_processor extracts the
// arguments between family-specific markers under strict JSON parsing
// and hands them back as a typed value.
//
// When the synthesized-tool path is unavailable (T is not a struct,
// chat construction failed, or the model declined to call the tool)
// the function falls through to a prompt-engineered fallback that
// augments the prompt with a JSON-shape instruction, generates, and
// runs extractJSON over the response.
//
// The C API does not currently expose a constrained-decoding schema
// delivery symbol (only the boolean toggle), so the strict-parse
// guarantee comes from the tool-call protocol, not from
// constrained-decoding at the runtime layer.

// defaultSchemaInstruction is a Printf format string with one %s
// placeholder for the shape hint, used by the fallback path. Imperative
// and short; explicitly forbids markdown fences.
const defaultSchemaInstruction = "Respond with valid JSON only — no commentary, no markdown fences.\n" +
	"The output must match this shape:\n%s"

// captureDirective prepends to the user's prompt on the tool-call
// path. Directs the model to deliver the answer as the synthesized
// tool's arguments rather than as free-form text.
const captureDirective = "Respond by calling the available tool with the structured value as its arguments. " +
	"Do not write any text outside the tool call.\n\n"

// GenerateData is the text-only convenience wrapper over
// GenerateDataMulti.
//
// On parse-path failure after the final attempt the caller receives a
// *GenerateDataError; use errors.As to inspect Phase / Raw / Attempts.
// Generate-phase errors (ctx cancellation, FFI failure) propagate
// immediately and do not trigger retries.
func GenerateData[T any](ctx context.Context, c *Client, prompt string, opts ...RuntimeOption) (*T, error) {
	return GenerateDataMulti[T](ctx, c, []Part{Text(prompt)}, opts...)
}

// GenerateDataMulti is the multimodal sibling of GenerateData. parts
// may carry text, image, and audio segments.
//
// Each attempt tries the synthesized-tool path first, then falls
// through to the prompt-engineered path if the tool-call did not
// deliver a value. Retries (WithRetries(n)) repeat the full sequence
// up to 1+n times. Generate-phase errors propagate immediately.
func GenerateDataMulti[T any](ctx context.Context, c *Client, parts []Part, opts ...RuntimeOption) (*T, error) {
	if c == nil {
		return nil, fmt.Errorf("litertlm: GenerateDataMulti: nil client")
	}
	cfg := resolveRuntimeConfig(opts)

	var lastErr error
	for attempt := 1; attempt <= 1+cfg.retries; attempt++ {
		if cerr := ctx.Err(); cerr != nil {
			return nil, cerr
		}

		if result := tryCaptureToolSilent[T](ctx, c, parts); result != nil {
			return result, nil
		}
		if cerr := ctx.Err(); cerr != nil {
			return nil, cerr
		}

		result, err := generateDataPromptEngineered[T](ctx, c, parts, cfg, attempt)
		if err == nil {
			return result, nil
		}
		var gdErr *GenerateDataError
		if errors.As(err, &gdErr) && gdErr.Phase == "generate" {
			return nil, err
		}
		lastErr = err
	}
	return nil, lastErr
}

// tryCaptureToolSilent runs the synthesized-tool capture path for T.
// Returns the captured value on success, nil otherwise. All failures
// (errCaptureToolUnsuitable, NewChat failure, transport error, the
// model declining to call the tool) are silent so the caller can fall
// through to the prompt-engineered path. ctx cancellation is observed
// via the caller's subsequent ctx.Err() check.
func tryCaptureToolSilent[T any](ctx context.Context, c *Client, parts []Part) *T {
	tool, err := getOrSynthesizeCaptureTool[T](c)
	if err != nil {
		return nil
	}

	var captured *T
	callCtx := context.WithValue(ctx, captureKey[T]{}, &captured)

	chat, err := c.NewChat(callCtx, WithTool(tool), WithMaxToolHops(1))
	if err != nil {
		return nil
	}
	defer func() { _ = chat.Close() }()

	if _, err := chat.SendMulti(callCtx, augmentForToolUse(parts)); err != nil {
		return nil
	}
	return captured
}

// generateDataPromptEngineered is the fallback: augment the prompt
// with a JSON-shape instruction, generate, and tolerantly parse the
// response. attempt is 1-indexed and used solely for error reporting;
// the outer retry loop in GenerateDataMulti drives iteration.
func generateDataPromptEngineered[T any](
	ctx context.Context, c *Client, parts []Part, cfg runtimeConfig, attempt int,
) (*T, error) {
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
		return nil, &GenerateDataError{
			Phase:    "parse",
			Err:      err,
			Raw:      text,
			Attempts: attempt,
		}
	}

	out := new(T)
	if err := json.Unmarshal([]byte(extracted), out); err != nil {
		return nil, &GenerateDataError{
			Phase:    "parse",
			Err:      err,
			Raw:      text,
			Attempts: attempt,
		}
	}
	return out, nil
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
// the LAST text Part. When no text part exists, Text(instruction) is
// appended at the end (works for image/audio-only inputs).
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

// augmentForToolUse returns a copy of parts with captureDirective
// prepended to the LAST text Part, mirroring injectSchema's placement
// rules so the directive lands close to the user's content. Appended
// as a new Text part when none exists.
func augmentForToolUse(parts []Part) []Part {
	out := append([]Part(nil), parts...)
	for i := len(out) - 1; i >= 0; i-- {
		if out[i].kind == partText {
			out[i].text = captureDirective + out[i].text
			return out
		}
	}
	return append(out, Text(captureDirective))
}
