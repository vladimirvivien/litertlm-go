package litertlm

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

// reservedToolNamePrefix is reserved for tools synthesized internally
// by the wrapper (e.g. GenerateData[T]'s capture tool). RegisterTool
// rejects user-supplied names that start with this prefix.
const reservedToolNamePrefix = "__litertlm_"

// ToolDefinition is the common contract every tool attached to a Chat
// satisfies. RawTool and ManagedTool[I, O] both implement it so they
// can be passed together to WithTool.
type ToolDefinition interface {
	Name() string
	Description() string
	Parameters() map[string]any
}

// dispatchable is implemented by *ManagedTool[I, O] regardless of its
// type parameters. Chat stores dispatchable entries in its registry;
// the auto-dispatch loop calls invoke through this interface and
// consults policy when invoke returns an error.
type dispatchable interface {
	invoke(ctx context.Context, argsJSON []byte) (any, error)
	policy() ToolPolicy
}

// ToolPolicy configures behaviors that apply to a single tool. Today
// it covers error handling; the type is named broadly so additional
// per-tool knobs can be added under the same option family.
type ToolPolicy int

const (
	// ToolPolicyReturnOnError (default) propagates a handler error
	// up as a Go error from Chat.Send; the dispatch loop ends and
	// the model is not informed.
	ToolPolicyReturnOnError ToolPolicy = iota

	// ToolPolicyInformOnError marshals the handler error's message
	// as the tool's result and continues the dispatch loop. The
	// model can apologize, retry with different arguments, or pick
	// a different tool.
	ToolPolicyInformOnError
)

// ToolOption configures a tool at RegisterTool time.
type ToolOption func(*toolConfig)

type toolConfig struct {
	policy ToolPolicy
}

// WithToolPolicy sets the tool's policy. See ToolPolicy for the
// available values.
func WithToolPolicy(p ToolPolicy) ToolOption {
	return func(c *toolConfig) { c.policy = p }
}

// RawTool is a hand-built tool declaration: a name, description, and
// a JSON-Schema parameters map. The model sees the declaration but
// the framework does not dispatch it — the caller observes
// Reply.ToolCalls and replies with Chat.SendToolResult on its own.
//
// Use ManagedTool (via RegisterTool) when the framework should run
// the handler and forward the result automatically.
type RawTool struct {
	name        string
	description string
	parameters  map[string]any
}

// NewRawTool constructs a RawTool from a name, description, and
// JSON-Schema parameters object. parameters follows the OpenAI /
// Anthropic function-calling shape (typically a top-level
// `{"type": "object", "properties": {...}, "required": [...]}`).
func NewRawTool(name, description string, parameters map[string]any) *RawTool {
	return &RawTool{
		name:        name,
		description: description,
		parameters:  parameters,
	}
}

// Name returns the tool's name.
func (r *RawTool) Name() string { return r.name }

// Description returns the tool's description.
func (r *RawTool) Description() string { return r.description }

// Parameters returns the JSON-Schema parameters object.
func (r *RawTool) Parameters() map[string]any { return r.parameters }

// ManagedTool wraps a typed handler and a JSON-Schema description of
// its input. Construct with RegisterTool; attach to a Chat with
// WithTool. The framework dispatches calls to the handler directly
// when the model invokes the tool.
type ManagedTool[I, O any] struct {
	name        string
	description string
	handler     func(context.Context, I) (O, error)
	parameters  map[string]any
	errPolicy   ToolPolicy
}

// Name returns the tool's name.
func (t *ManagedTool[I, O]) Name() string { return t.name }

// Description returns the tool's description.
func (t *ManagedTool[I, O]) Description() string { return t.description }

// Parameters returns the JSON-Schema parameters object derived from I.
func (t *ManagedTool[I, O]) Parameters() map[string]any { return t.parameters }

// policy returns the configured ToolPolicy. Used by the dispatch
// loop to decide what to do with a handler error.
func (t *ManagedTool[I, O]) policy() ToolPolicy { return t.errPolicy }

// invoke unmarshals argsJSON into a fresh I, runs the handler, and
// returns the typed O. The dispatcher marshals the result back to
// JSON for the model.
func (t *ManagedTool[I, O]) invoke(ctx context.Context, argsJSON []byte) (any, error) {
	var in I
	if len(argsJSON) > 0 && string(argsJSON) != "null" {
		if err := json.Unmarshal(argsJSON, &in); err != nil {
			return nil, fmt.Errorf("litertlm: tool %q: unmarshal args: %w", t.name, err)
		}
	}
	out, err := t.handler(ctx, in)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// RegisterTool builds a ManagedTool[I, O] and registers it on c so
// the same name cannot be reused. I must be a struct or pointer to a
// struct; the returned tool exposes a JSON-Schema parameters map
// derived from I's exported fields.
//
// Field rules for I:
//   - Exported fields only.
//   - Field name from `json:"name"` tag, else lowercased Go name.
//     `json:"-"` excludes the field.
//   - Description from `description:"..."` tag, optional.
//   - Pointer fields are optional; non-pointer fields are required.
//   - Supported kinds: string, bool, all int/uint/float widths,
//     slice/array, nested struct, pointer (unwrapped).
//   - Nesting capped at depth 32.
//
// Attach the returned tool to a Chat with WithTool(t). Pass
// WithToolPolicy to override the default error-handling policy.
func RegisterTool[I, O any](
	c *Client,
	name, description string,
	handler func(context.Context, I) (O, error),
	opts ...ToolOption,
) (*ManagedTool[I, O], error) {
	if c == nil {
		return nil, fmt.Errorf("litertlm: RegisterTool: nil client")
	}
	if name == "" {
		return nil, fmt.Errorf("litertlm: RegisterTool: empty name")
	}
	if strings.HasPrefix(name, reservedToolNamePrefix) {
		return nil, fmt.Errorf("litertlm: RegisterTool %q: names starting with %q are reserved for wrapper-internal tools",
			name, reservedToolNamePrefix)
	}
	if handler == nil {
		return nil, fmt.Errorf("litertlm: RegisterTool %q: nil handler", name)
	}

	params, err := paramsSchemaOf(reflect.TypeFor[I]())
	if err != nil {
		return nil, fmt.Errorf("litertlm: RegisterTool %q: %w", name, err)
	}

	cfg := toolConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}

	tool := &ManagedTool[I, O]{
		name:        name,
		description: description,
		handler:     handler,
		parameters:  params,
		errPolicy:   cfg.policy,
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.tools == nil {
		c.tools = map[string]ToolDefinition{}
	}
	if _, exists := c.tools[name]; exists {
		return nil, fmt.Errorf("litertlm: RegisterTool: tool %q already registered", name)
	}
	c.tools[name] = tool
	return tool, nil
}

// unregisterTool removes name from the client's tool registry. Used
// by internal synthesized-tool flows that need cleanup paths. No-op
// when name is absent.
func (c *Client) unregisterTool(name string) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.tools, name)
}

// paramsSchemaOf generates a formal JSON-Schema object for type t.
// t must be a struct or pointer to a struct. The output shape
// matches the OpenAI / Anthropic function-calling parameters
// convention:
//
//	{
//	    "type":       "object",
//	    "properties": {<field>: <schema>, ...},
//	    "required":   ["<field>", ...]
//	}
func paramsSchemaOf(t reflect.Type) (map[string]any, error) {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("paramsSchemaOf: type %s must be a struct or pointer to struct", t)
	}
	return paramsSchemaStruct(t, 0)
}

func paramsSchemaStruct(t reflect.Type, depth int) (map[string]any, error) {
	if depth > 32 {
		return nil, fmt.Errorf("paramsSchemaOf: type %s nests deeper than 32 levels", t)
	}

	properties := map[string]any{}
	var required []string

	for f := range t.Fields() {
		f := f
		if !f.IsExported() {
			continue
		}
		name := jsonFieldName(f)
		if name == "-" {
			continue
		}

		fieldSchema, err := paramsSchemaForType(f.Type, depth+1)
		if err != nil {
			return nil, err
		}
		if desc := f.Tag.Get("description"); desc != "" {
			fieldSchema["description"] = desc
		}
		properties[name] = fieldSchema

		if f.Type.Kind() != reflect.Pointer {
			required = append(required, name)
		}
	}

	schema := map[string]any{
		"type":       "object",
		"properties": properties,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema, nil
}

func paramsSchemaForType(t reflect.Type, depth int) (map[string]any, error) {
	if depth > 32 {
		return nil, fmt.Errorf("paramsSchemaOf: type %s nests deeper than 32 levels", t)
	}
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}, nil
	case reflect.Bool:
		return map[string]any{"type": "boolean"}, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]any{"type": "integer"}, nil
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}, nil
	case reflect.Slice, reflect.Array:
		items, err := paramsSchemaForType(t.Elem(), depth+1)
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "array", "items": items}, nil
	case reflect.Struct:
		return paramsSchemaStruct(t, depth+1)
	default:
		return nil, fmt.Errorf("paramsSchemaOf: unsupported kind %s for type %s", t.Kind(), t)
	}
}
