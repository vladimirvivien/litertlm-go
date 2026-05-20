package litertlm

import (
	"context"
	"errors"
	"fmt"
	"hash/fnv"
	"reflect"
)

// captureKey[T] is the context.Value key the synthesized capture tool
// reads at dispatch time to find the per-call destination pointer.
// Generic over T so distinct types in the same Client don't collide.
type captureKey[T any] struct{}

// errCaptureToolUnsuitable signals that T cannot be expressed as a
// JSON-Schema object (e.g. T is a slice, map, or scalar). The
// synthesized-tool path is unavailable; the caller falls through to
// the prompt-engineered path.
var errCaptureToolUnsuitable = errors.New("litertlm: capture tool unsuitable for type")

// captureToolName returns a deterministic, reserved-prefixed tool name
// derived from T's reflect.Type.String(). The hash is FNV-1a 64-bit;
// the name is short, alphanumeric, and stable across runs.
func captureToolName[T any]() string {
	h := fnv.New64a()
	h.Write([]byte(reflect.TypeFor[T]().String()))
	return fmt.Sprintf("%scapture_%016x", reservedToolNamePrefix, h.Sum64())
}

// captureDescription is the model-facing tool description. The wording
// is a direct instruction so the model invokes the tool exactly once
// with the structured value as its arguments.
func captureDescription[T any]() string {
	return fmt.Sprintf(
		"Deliver the requested structured value as the arguments to this tool. "+
			"Call this tool exactly once. Do not call any other tool. "+
			"Do not emit any text outside the tool call. "+
			"The arguments object conforms to the JSON-Schema for type %s.",
		reflect.TypeFor[T]().String())
}

// getOrSynthesizeCaptureTool returns the cached capture tool for T,
// registering it on first call. The handler reads the per-call
// destination from ctx.Value(captureKey[T]{}) and writes the typed
// input into it.
//
// Returns errCaptureToolUnsuitable when T is not a struct (and thus
// has no JSON-Schema object representation). Callers should treat
// that sentinel as a signal to fall back to the prompt-engineered
// GenerateData path.
func getOrSynthesizeCaptureTool[T any](c *Client) (*ManagedTool[T, struct{}], error) {
	if c == nil {
		return nil, fmt.Errorf("litertlm: getOrSynthesizeCaptureTool: nil client")
	}

	t := reflect.TypeFor[T]()
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("%w: %s", errCaptureToolUnsuitable, reflect.TypeFor[T]().String())
	}

	name := captureToolName[T]()

	c.mu.Lock()
	if existing, ok := c.tools[name]; ok {
		c.mu.Unlock()
		typed, ok := existing.(*ManagedTool[T, struct{}])
		if !ok {
			return nil, fmt.Errorf("litertlm: capture tool slot %q occupied by incompatible type", name)
		}
		return typed, nil
	}
	c.mu.Unlock()

	params, err := paramsSchemaOf(reflect.TypeFor[T]())
	if err != nil {
		return nil, fmt.Errorf("%w: %s: %v", errCaptureToolUnsuitable, reflect.TypeFor[T]().String(), err)
	}

	tool := &ManagedTool[T, struct{}]{
		name:        name,
		description: captureDescription[T](),
		handler:     captureHandler[T](),
		parameters:  params,
		errPolicy:   ToolPolicyReturnOnError,
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if existing, ok := c.tools[name]; ok {
		typed, ok := existing.(*ManagedTool[T, struct{}])
		if !ok {
			return nil, fmt.Errorf("litertlm: capture tool slot %q occupied by incompatible type", name)
		}
		return typed, nil
	}
	if c.tools == nil {
		c.tools = map[string]ToolDefinition{}
	}
	c.tools[name] = tool
	return tool, nil
}

// captureHandler is the shared closure all GenerateData[T] calls of a
// given T route through. At dispatch time it pulls the per-call
// destination pointer from ctx.Value and writes the typed input.
// Concurrent calls with distinct contexts each see their own slot.
func captureHandler[T any]() func(context.Context, T) (struct{}, error) {
	return func(ctx context.Context, in T) (struct{}, error) {
		slot, ok := ctx.Value(captureKey[T]{}).(**T)
		if !ok || slot == nil {
			return struct{}{}, nil
		}
		v := in
		*slot = &v
		return struct{}{}, nil
	}
}
