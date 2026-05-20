package litertlm

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

// ---- paramsSchemaOf ------------------------------------------------------

type primInput struct {
	Name   string  `description:"a name"`
	Age    int     `description:"an age"`
	Score  float64 `description:"a score"`
	Active bool
}

func TestParamsSchemaOf_Primitives(t *testing.T) {
	got, err := paramsSchemaOf(reflect.TypeFor[primInput]())
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	want := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name":   map[string]any{"type": "string", "description": "a name"},
			"age":    map[string]any{"type": "integer", "description": "an age"},
			"score":  map[string]any{"type": "number", "description": "a score"},
			"active": map[string]any{"type": "boolean"},
		},
		"required": []string{"name", "age", "score", "active"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("schema mismatch\n got: %#v\nwant: %#v", got, want)
	}
}

type taggedInput struct {
	Title string `json:"display_title" description:"title shown in UI"`
	Skip  string `json:"-"`
	Plain string
}

func TestParamsSchemaOf_JSONTagsAndSkip(t *testing.T) {
	got, err := paramsSchemaOf(reflect.TypeFor[taggedInput]())
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	props := got["properties"].(map[string]any)
	if _, ok := props["display_title"]; !ok {
		t.Errorf("expected json-tagged name 'display_title', got %v", props)
	}
	if _, ok := props["title"]; ok {
		t.Errorf("Go field name 'title' should not appear when json tag is set")
	}
	if _, ok := props["skip"]; ok {
		t.Errorf("json:\"-\" field should be omitted")
	}
	if _, ok := props["plain"]; !ok {
		t.Errorf("untagged field 'plain' should appear")
	}
	required := got["required"].([]string)
	for _, name := range required {
		if name == "skip" {
			t.Errorf("skipped field appeared in required[]")
		}
	}
}

type sliceInput struct {
	Tags    []string
	Numbers []int `description:"a list of numbers"`
}

func TestParamsSchemaOf_Slice(t *testing.T) {
	got, err := paramsSchemaOf(reflect.TypeFor[sliceInput]())
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	props := got["properties"].(map[string]any)
	tags := props["tags"].(map[string]any)
	if tags["type"] != "array" {
		t.Errorf("tags.type = %v, want array", tags["type"])
	}
	items := tags["items"].(map[string]any)
	if items["type"] != "string" {
		t.Errorf("tags.items.type = %v, want string", items["type"])
	}
	numbers := props["numbers"].(map[string]any)
	if numbers["description"] != "a list of numbers" {
		t.Errorf("description not propagated to slice field")
	}
}

type nestedInput struct {
	User userProfile
}

type userProfile struct {
	Name string `description:"user name"`
	Age  int
}

func TestParamsSchemaOf_NestedStruct(t *testing.T) {
	got, err := paramsSchemaOf(reflect.TypeFor[nestedInput]())
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	user := got["properties"].(map[string]any)["user"].(map[string]any)
	if user["type"] != "object" {
		t.Errorf("user.type = %v, want object", user["type"])
	}
	userProps := user["properties"].(map[string]any)
	name := userProps["name"].(map[string]any)
	if name["description"] != "user name" {
		t.Errorf("nested description not preserved")
	}
	required := user["required"].([]string)
	if len(required) != 2 {
		t.Errorf("nested required = %v, want 2 entries", required)
	}
}

type optionalInput struct {
	Required string
	Optional *string `description:"optional note"`
	OptInt   *int
}

func TestParamsSchemaOf_PointerOptional(t *testing.T) {
	got, err := paramsSchemaOf(reflect.TypeFor[optionalInput]())
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	required := got["required"].([]string)
	if len(required) != 1 || required[0] != "required" {
		t.Errorf("required = %v, want [required] only (pointer fields optional)", required)
	}
	props := got["properties"].(map[string]any)
	optional := props["optional"].(map[string]any)
	if optional["type"] != "string" {
		t.Errorf("pointer field unwrap failed; got %v", optional)
	}
	if optional["description"] != "optional note" {
		t.Errorf("description on pointer field not preserved")
	}
}

func TestParamsSchemaOf_RejectsNonStruct(t *testing.T) {
	_, err := paramsSchemaOf(reflect.TypeFor[string]())
	if err == nil {
		t.Fatal("expected error for non-struct type")
	}
}

func TestParamsSchemaOf_AcceptsPointerToStruct(t *testing.T) {
	got, err := paramsSchemaOf(reflect.TypeFor[*primInput]())
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if got["type"] != "object" {
		t.Errorf("got %v, want object", got["type"])
	}
}

type unsupportedInput struct {
	Ch chan int
}

func TestParamsSchemaOf_UnsupportedKind(t *testing.T) {
	_, err := paramsSchemaOf(reflect.TypeFor[unsupportedInput]())
	if err == nil {
		t.Fatal("expected error for unsupported kind")
	}
}

// ---- RawTool ------------------------------------------------------------

func TestNewRawTool(t *testing.T) {
	params := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"q": map[string]any{"type": "string"},
		},
	}
	r := NewRawTool("search", "search the web", params)
	if r.Name() != "search" {
		t.Errorf("Name() = %q", r.Name())
	}
	if r.Description() != "search the web" {
		t.Errorf("Description() = %q", r.Description())
	}
	if !reflect.DeepEqual(r.Parameters(), params) {
		t.Errorf("Parameters() mismatch")
	}
	// Compile-time + runtime check that *RawTool satisfies ToolDefinition.
	var _ ToolDefinition = r
}

// ---- RegisterTool -------------------------------------------------------

type weatherIn struct {
	Location string `description:"city and state"`
}
type weatherOut struct {
	Forecast string `json:"forecast"`
}

func TestRegisterTool_Basic(t *testing.T) {
	c := &Client{}
	tool, err := RegisterTool(c, "get_weather", "fetch a weather forecast",
		func(ctx context.Context, in weatherIn) (weatherOut, error) {
			return weatherOut{Forecast: "sunny in " + in.Location}, nil
		})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if tool.Name() != "get_weather" {
		t.Errorf("Name() = %q", tool.Name())
	}
	if tool.Description() != "fetch a weather forecast" {
		t.Errorf("Description() = %q", tool.Description())
	}
	params := tool.Parameters()
	if params["type"] != "object" {
		t.Errorf("Parameters().type = %v, want object", params["type"])
	}
	if got, ok := c.tools["get_weather"]; !ok || got != ToolDefinition(tool) {
		t.Errorf("tool not registered on client")
	}
}

func TestRegisterTool_DuplicateName(t *testing.T) {
	c := &Client{}
	handler := func(ctx context.Context, in weatherIn) (weatherOut, error) {
		return weatherOut{}, nil
	}
	if _, err := RegisterTool(c, "dup", "first", handler); err != nil {
		t.Fatalf("first RegisterTool: %v", err)
	}
	if _, err := RegisterTool(c, "dup", "second", handler); err == nil {
		t.Error("expected error on duplicate name")
	}
}

func TestRegisterTool_NilArgs(t *testing.T) {
	handler := func(ctx context.Context, in weatherIn) (weatherOut, error) {
		return weatherOut{}, nil
	}
	if _, err := RegisterTool[weatherIn, weatherOut](nil, "x", "d", handler); err == nil {
		t.Error("expected error for nil client")
	}
	c := &Client{}
	if _, err := RegisterTool(c, "", "d", handler); err == nil {
		t.Error("expected error for empty name")
	}
	if _, err := RegisterTool[weatherIn, weatherOut](c, "x", "d", nil); err == nil {
		t.Error("expected error for nil handler")
	}
}

func TestRegisterTool_RejectsNonStructInput(t *testing.T) {
	c := &Client{}
	if _, err := RegisterTool(c, "bad", "d",
		func(ctx context.Context, s string) (string, error) { return s, nil }); err == nil {
		t.Error("expected error when I is not a struct")
	}
}

func TestRegisterTool_RejectsReservedPrefix(t *testing.T) {
	c := &Client{}
	handler := func(ctx context.Context, in weatherIn) (weatherOut, error) {
		return weatherOut{}, nil
	}
	cases := []string{
		"__litertlm_",
		"__litertlm_capture",
		"__litertlm_capture_Person",
	}
	for _, name := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := RegisterTool(c, name, "d", handler)
			if err == nil {
				t.Fatalf("RegisterTool(%q) succeeded; want reserved-prefix error", name)
			}
			if got, ok := c.tools[name]; ok {
				t.Errorf("rejected tool was still stored: %v", got)
			}
		})
	}
}

func TestRegisterTool_AcceptsNamesContainingReservedPrefix(t *testing.T) {
	c := &Client{}
	handler := func(ctx context.Context, in weatherIn) (weatherOut, error) {
		return weatherOut{}, nil
	}
	if _, err := RegisterTool(c, "my__litertlm_thing", "d", handler); err != nil {
		t.Fatalf("RegisterTool: %v — prefix rejection must match start-of-name only", err)
	}
}

// ---- unregisterTool -----------------------------------------------------

func TestUnregisterTool_RemovesEntry(t *testing.T) {
	c := &Client{}
	handler := func(ctx context.Context, in weatherIn) (weatherOut, error) {
		return weatherOut{}, nil
	}
	if _, err := RegisterTool(c, "weather", "d", handler); err != nil {
		t.Fatalf("RegisterTool: %v", err)
	}
	c.unregisterTool("weather")
	if _, ok := c.tools["weather"]; ok {
		t.Error("unregisterTool: entry still present")
	}
}

func TestUnregisterTool_NoOpForMissing(t *testing.T) {
	c := &Client{}
	c.unregisterTool("never_registered") // must not panic
	if len(c.tools) != 0 {
		t.Errorf("unregisterTool created entries: %v", c.tools)
	}
}

func TestUnregisterTool_NilClient(t *testing.T) {
	var c *Client
	c.unregisterTool("anything") // must not panic
}

// ---- ManagedTool.invoke -------------------------------------------------

func TestManagedTool_Invoke(t *testing.T) {
	c := &Client{}
	tool, err := RegisterTool(c, "echo", "echoes location",
		func(ctx context.Context, in weatherIn) (weatherOut, error) {
			return weatherOut{Forecast: in.Location}, nil
		})
	if err != nil {
		t.Fatalf("register: %v", err)
	}

	out, err := tool.invoke(context.Background(), []byte(`{"location":"Boston, MA"}`))
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	res, ok := out.(weatherOut)
	if !ok {
		t.Fatalf("expected weatherOut, got %T", out)
	}
	if res.Forecast != "Boston, MA" {
		t.Errorf("Forecast = %q", res.Forecast)
	}
}

func TestManagedTool_InvokeMalformedJSON(t *testing.T) {
	c := &Client{}
	tool, err := RegisterTool(c, "echo", "",
		func(ctx context.Context, in weatherIn) (weatherOut, error) {
			return weatherOut{}, nil
		})
	if err != nil {
		t.Fatalf("register: %v", err)
	}
	_, err = tool.invoke(context.Background(), []byte(`{not json`))
	if err == nil {
		t.Error("expected unmarshal error")
	}
}

func TestManagedTool_InvokePropagatesHandlerError(t *testing.T) {
	wantErr := errors.New("boom")
	c := &Client{}
	tool, err := RegisterTool(c, "fail", "",
		func(ctx context.Context, in weatherIn) (weatherOut, error) {
			return weatherOut{}, wantErr
		})
	if err != nil {
		t.Fatalf("register: %v", err)
	}
	_, err = tool.invoke(context.Background(), []byte(`{"location":"x"}`))
	if !errors.Is(err, wantErr) {
		t.Errorf("err = %v, want %v", err, wantErr)
	}
}

// ---- buildToolRegistry --------------------------------------------------

func TestBuildToolRegistry_MixedRawAndManaged(t *testing.T) {
	c := &Client{}
	managed, _ := RegisterTool(c, "lookup", "",
		func(ctx context.Context, in weatherIn) (weatherOut, error) {
			return weatherOut{}, nil
		})
	raw := NewRawTool("manual", "hand-built", map[string]any{"type": "object"})

	reg, err := buildToolRegistry([]ToolDefinition{raw, managed})
	if err != nil {
		t.Fatalf("buildToolRegistry: %v", err)
	}
	if reg["manual"] != ToolDefinition(raw) {
		t.Errorf("registry missing raw entry")
	}
	if reg["lookup"] != ToolDefinition(managed) {
		t.Errorf("registry missing managed entry")
	}
}

func TestBuildToolRegistry_DuplicateName(t *testing.T) {
	a := NewRawTool("dup", "", nil)
	b := NewRawTool("dup", "", nil)
	if _, err := buildToolRegistry([]ToolDefinition{a, b}); err == nil {
		t.Error("expected duplicate-name error")
	}
}

func TestBuildToolRegistry_EmptyName(t *testing.T) {
	if _, err := buildToolRegistry([]ToolDefinition{NewRawTool("", "", nil)}); err == nil {
		t.Error("expected empty-name error")
	}
}
