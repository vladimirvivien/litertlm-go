package litertlm

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
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
	// as the tool's result and continues the dispatch loop.
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
type RawTool struct {
	name        string
	description string
	parameters  map[string]any
}

// NewRawTool constructs a RawTool from a name, description, and
// JSON-Schema parameters object. The parameters map is cloned so later
// caller mutations do not affect the tool declaration.
func NewRawTool(name, description string, parameters map[string]any) *RawTool {
	return &RawTool{
		name:        name,
		description: description,
		parameters:  cloneJSONMap(parameters),
	}
}

// Name returns the tool's name.
func (r *RawTool) Name() string { return r.name }

// Description returns the tool's description.
func (r *RawTool) Description() string { return r.description }

// Parameters returns a defensive copy of the JSON-Schema parameters object.
func (r *RawTool) Parameters() map[string]any { return cloneJSONMap(r.parameters) }

// ManagedTool wraps a typed handler and a JSON-Schema description of
// its input. Construct with RegisterTool; attach to a Chat with WithTool.
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

// Parameters returns a defensive copy of the JSON-Schema parameters object.
func (t *ManagedTool[I, O]) Parameters() map[string]any { return cloneJSONMap(t.parameters) }

func (t *ManagedTool[I, O]) policy() ToolPolicy { return t.errPolicy }

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
// the same name cannot be reused.
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

func cloneJSONMap(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = cloneJSONValue(v)
	}
	return out
}

func cloneJSONValue(v any) any {
	switch x := v.(type) {
	case map[string]any:
		return cloneJSONMap(x)
	case []any:
		cp := make([]any, len(x))
		for i, elem := range x {
			cp[i] = cloneJSONValue(elem)
		}
		return cp
	case []string:
		return slices.Clone(x)
	case []int:
		return slices.Clone(x)
	default:
		return v
	}
}

// paramsSchemaOf generates a formal JSON-Schema object for type t.
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
