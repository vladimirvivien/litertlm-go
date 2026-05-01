package litertlm

import "testing"

// TestReply_TextOnly covers the simple chat-reply shape:
//
//	{"role":"assistant","content":[{"type":"text","text":"hi"}]}
func TestReply_TextOnly(t *testing.T) {
	raw := `{"role":"assistant","content":[{"type":"text","text":"hello"}]}`
	r, err := parseReply(raw)
	if err != nil {
		t.Fatalf("parseReply: %v", err)
	}
	if r.Text() != "hello" {
		t.Errorf("Text = %q, want hello", r.Text())
	}
	if r.HasToolCalls() {
		t.Errorf("HasToolCalls = true, want false")
	}
	if r.Raw() != raw {
		t.Errorf("Raw round-trip differs")
	}
}

// TestReply_MultipleTextParts concatenates every text part.
func TestReply_MultipleTextParts(t *testing.T) {
	raw := `{"role":"assistant","content":[
		{"type":"text","text":"foo "},
		{"type":"text","text":"bar"}
	]}`
	r, _ := parseReply(raw)
	if r.Text() != "foo bar" {
		t.Errorf("Text = %q, want \"foo bar\"", r.Text())
	}
}

// TestReply_NonTextPartsIgnored skips image/audio parts in Text().
func TestReply_NonTextPartsIgnored(t *testing.T) {
	raw := `{"role":"assistant","content":[
		{"type":"image"},
		{"type":"text","text":"caption"},
		{"type":"audio"}
	]}`
	r, _ := parseReply(raw)
	if r.Text() != "caption" {
		t.Errorf("Text = %q, want caption", r.Text())
	}
}

// TestReply_ToolCalls covers the structured-tool-call shape:
//
//	{"role":"assistant","tool_calls":[{...}]}
func TestReply_ToolCalls(t *testing.T) {
	raw := `{"role":"assistant","tool_calls":[
		{"type":"function","function":{"name":"calc_add","arguments":{"a":17.0,"b":25.0}}}
	]}`
	r, err := parseReply(raw)
	if err != nil {
		t.Fatalf("parseReply: %v", err)
	}
	if !r.HasToolCalls() {
		t.Errorf("HasToolCalls = false, want true")
	}
	tc := r.ToolCalls()
	if len(tc) != 1 || tc[0].Function.Name != "calc_add" {
		t.Fatalf("tool_calls parse failure: %+v", tc)
	}
	// Numeric args come through as float64 from encoding/json.
	if a, ok := tc[0].Function.Arguments["a"].(float64); !ok || a != 17 {
		t.Errorf("arg a = %v (%T), want 17.0", tc[0].Function.Arguments["a"], tc[0].Function.Arguments["a"])
	}
}

// TestReply_QuoteMarkerStripped ensures Gemma 4's `<|"|>` markers are
// removed from string-typed tool-call arguments.
func TestReply_QuoteMarkerStripped(t *testing.T) {
	raw := `{"role":"assistant","tool_calls":[
		{"type":"function","function":{"name":"get_weather","arguments":{"location":"<|\"|>Boston, MA<|\"|>"}}}
	]}`
	r, _ := parseReply(raw)
	loc := r.ToolCalls()[0].Function.Arguments["location"]
	if loc != "Boston, MA" {
		t.Errorf("location = %q, want \"Boston, MA\" (markers should be stripped)", loc)
	}
}

// TestReply_QuoteMarkerOnlyStringArgs ensures the marker-strip pass
// doesn't blow up on numeric args (they aren't strings, so .(string)
// type-assert returns false).
func TestReply_QuoteMarkerOnlyStringArgs(t *testing.T) {
	raw := `{"role":"assistant","tool_calls":[
		{"type":"function","function":{"name":"calc_add","arguments":{"a":17.0,"b":25.0}}}
	]}`
	r, err := parseReply(raw)
	if err != nil {
		t.Fatalf("parseReply: %v", err)
	}
	if r.ToolCalls()[0].Function.Arguments["a"].(float64) != 17 {
		t.Errorf("numeric arg corrupted by marker-strip pass")
	}
}

// TestReply_Malformed surfaces a parse error rather than panicking.
func TestReply_Malformed(t *testing.T) {
	_, err := parseReply(`not json`)
	if err == nil {
		t.Fatal("expected parse error on non-JSON input")
	}
}

// TestReply_NilMethods are safe on a nil *Reply (defensive — callers
// shouldn't hit this path, but we shouldn't panic if they do).
func TestReply_NilMethods(t *testing.T) {
	var r *Reply
	if r.Text() != "" {
		t.Errorf("nil.Text() should be empty")
	}
	if r.HasToolCalls() {
		t.Errorf("nil.HasToolCalls() should be false")
	}
	if r.ToolCalls() != nil {
		t.Errorf("nil.ToolCalls() should be nil")
	}
	if r.Raw() != "" {
		t.Errorf("nil.Raw() should be empty")
	}
}
