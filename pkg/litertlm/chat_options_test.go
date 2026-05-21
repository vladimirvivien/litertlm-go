package litertlm

import (
	"strings"
	"testing"
)

func TestChatOption_Defaults(t *testing.T) {
	cfg := chatConfig{}
	if cfg.systemPromptSet {
		t.Errorf("systemPromptSet should default false")
	}
	if cfg.tools != nil {
		t.Errorf("tools should default nil")
	}
	if cfg.initialMessages != nil {
		t.Errorf("initialMessages should default nil")
	}
	if cfg.constrainedDecoding {
		t.Errorf("constrainedDecoding should default false")
	}
}

func TestChatOption_SystemPrompt(t *testing.T) {
	cfg := chatConfig{}
	WithSystemPrompt("you are helpful")(&cfg)
	if !cfg.systemPromptSet || cfg.systemPrompt != "you are helpful" {
		t.Errorf("systemPrompt = %q (set=%v)", cfg.systemPrompt, cfg.systemPromptSet)
	}

	// Empty string should still set the flag — useful if a caller wants
	// to explicitly clear the system prompt.
	cfg = chatConfig{}
	WithSystemPrompt("")(&cfg)
	if !cfg.systemPromptSet {
		t.Errorf("WithSystemPrompt(\"\") should still set the flag")
	}
}

func TestChatOption_WithExtraContext(t *testing.T) {
	cfg := chatConfig{}
	if cfg.extraContextJSON != "" {
		t.Fatalf("extraContextJSON should default empty")
	}
	WithExtraContext(`{"weather":"sunny"}`)(&cfg)
	if cfg.extraContextJSON != `{"weather":"sunny"}` {
		t.Errorf("extraContextJSON = %q", cfg.extraContextJSON)
	}
}

func TestChatOption_WithFilterChannelContentFromKVCache(t *testing.T) {
	cfg := chatConfig{}
	if cfg.filterChannelContentFromKVCache != nil {
		t.Fatalf("filterChannelContentFromKVCache should default nil")
	}
	WithFilterChannelContentFromKVCache(true)(&cfg)
	if cfg.filterChannelContentFromKVCache == nil || !*cfg.filterChannelContentFromKVCache {
		t.Errorf("filterChannelContentFromKVCache = %v", cfg.filterChannelContentFromKVCache)
	}
	WithFilterChannelContentFromKVCache(false)(&cfg)
	if cfg.filterChannelContentFromKVCache == nil || *cfg.filterChannelContentFromKVCache {
		t.Errorf("toggle to false failed: %v", cfg.filterChannelContentFromKVCache)
	}
}

func TestChatOption_WithToolAccumulates(t *testing.T) {
	a := NewRawTool("a", "", nil)
	b := NewRawTool("b", "", nil)
	cfg := chatConfig{}
	WithTool(a)(&cfg)
	WithTool(b)(&cfg)

	if len(cfg.tools) != 2 {
		t.Fatalf("tools = %d, want 2", len(cfg.tools))
	}
	if cfg.tools[0].Name() != "a" || cfg.tools[1].Name() != "b" {
		t.Errorf("tool order broken: got %q, %q", cfg.tools[0].Name(), cfg.tools[1].Name())
	}
}

func TestChatOption_InitialMessagesCopy(t *testing.T) {
	msgs := []Message{{Role: "user", Parts: []Part{Text("hi")}}}
	cfg := chatConfig{}
	WithInitialMessages(msgs)(&cfg)
	msgs[0].Role = "mutated"
	if cfg.initialMessages[0].Role != "user" {
		t.Errorf("expected defensive copy of the outer slice")
	}
}

func TestChatOption_EncodeSystemPromptBareContent(t *testing.T) {
	// The C-API contract demands bare content (a JSON-encoded string),
	// not a {role,content} envelope. encodeSystemPrompt must produce a
	// plain JSON string (e.g. `"hello"`), not an object.
	cfg := chatConfig{}
	WithSystemPrompt("hello")(&cfg)
	got, err := encodeSystemPrompt(cfg)
	if err != nil {
		t.Fatalf("encodeSystemPrompt: %v", err)
	}
	if got != `"hello"` {
		t.Errorf("encodeSystemPrompt = %q, want \"hello\"", got)
	}
	if strings.Contains(got, "role") || strings.Contains(got, "content") {
		t.Errorf("system prompt should be bare content, not an envelope: %q", got)
	}
}

func TestChatOption_EncodeSystemPromptUnset(t *testing.T) {
	got, err := encodeSystemPrompt(chatConfig{})
	if err != nil || got != "" {
		t.Errorf("unset system prompt: got %q err=%v, want empty/nil", got, err)
	}
}

// TestEncodeMessages_TextOnly verifies a pure-text history seeds as
// {role, content:[{type:"text",text:...}]} per message.
func TestEncodeMessages_TextOnly(t *testing.T) {
	msgs := []Message{
		{Role: "user", Parts: []Part{Text("hello")}},
		{Role: "assistant", Parts: []Part{Text("hi back")}},
	}
	got, err := encodeMessages(msgs)
	if err != nil {
		t.Fatalf("encodeMessages: %v", err)
	}
	wantSubs := []string{
		`"role":"user"`, `"text":"hello"`,
		`"role":"assistant"`, `"text":"hi back"`,
		`"type":"text"`,
	}
	for _, s := range wantSubs {
		if !strings.Contains(got, s) {
			t.Errorf("encoded message missing %q in %s", s, got)
		}
	}
}

// TestEncodeMessages_Multimodal verifies a history mixing text and
// image parts emits both content entry types and base64-encodes the
// image bytes.
func TestEncodeMessages_Multimodal(t *testing.T) {
	imgBytes := []byte{0x89, 'P', 'N', 'G'}
	msgs := []Message{
		{Role: "user", Parts: []Part{Image(imgBytes), Text("what is this?")}},
		{Role: "assistant", Parts: []Part{Text("a PNG header")}},
	}
	got, err := encodeMessages(msgs)
	if err != nil {
		t.Fatalf("encodeMessages: %v", err)
	}
	wantSubs := []string{
		`"type":"image"`,
		`"blob":"iVBORw=="`, // base64 of 0x89 'P' 'N' 'G'
		`"type":"text"`,
		`"text":"what is this?"`,
		`"role":"assistant"`,
	}
	for _, s := range wantSubs {
		if !strings.Contains(got, s) {
			t.Errorf("encoded message missing %q in %s", s, got)
		}
	}
}

// TestEncodeMessages_EmptyRoleRejected verifies an empty Role is
// rejected up front rather than letting the C side fail mid-render.
func TestEncodeMessages_EmptyRoleRejected(t *testing.T) {
	msgs := []Message{{Role: "", Parts: []Part{Text("orphan")}}}
	_, err := encodeMessages(msgs)
	if err == nil {
		t.Fatal("expected error for empty Role")
	}
}

func TestChatOption_EncodeToolsArray(t *testing.T) {
	defs := []ToolDefinition{NewRawTool("calc_add", "Add two ints", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"a": map[string]any{"type": "integer"},
			"b": map[string]any{"type": "integer"},
		},
	})}
	got, err := encodeTools(defs)
	if err != nil {
		t.Fatalf("encodeTools: %v", err)
	}
	if !strings.HasPrefix(got, "[") || !strings.HasSuffix(got, "]") {
		t.Errorf("encodeTools should produce a JSON array: %q", got)
	}
	if !strings.Contains(got, `"calc_add"`) {
		t.Errorf("expected function name in encoded tools: %q", got)
	}
	if !strings.Contains(got, `"type":"function"`) {
		t.Errorf("expected `type:function` envelope: %q", got)
	}
}
