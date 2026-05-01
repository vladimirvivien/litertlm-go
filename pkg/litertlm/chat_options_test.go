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

func TestChatOption_ToolsCopy(t *testing.T) {
	t1 := Tool{Type: "function", Function: ToolFunction{Name: "a"}}
	tools := []Tool{t1}
	cfg := chatConfig{}
	WithTools(tools)(&cfg)

	if len(cfg.tools) != 1 || cfg.tools[0].Function.Name != "a" {
		t.Fatalf("tools not copied: %+v", cfg.tools)
	}

	// Mutate caller's slice; the chatConfig copy must be unaffected.
	tools[0].Function.Name = "mutated"
	if cfg.tools[0].Function.Name != "a" {
		t.Errorf("expected chatConfig.tools to be a defensive copy")
	}
}

func TestChatOption_InitialMessagesCopy(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "hi"}}
	cfg := chatConfig{}
	WithInitialMessages(msgs)(&cfg)
	msgs[0].Content = "mutated"
	if cfg.initialMessages[0].Content != "hi" {
		t.Errorf("expected defensive copy")
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

func TestChatOption_EncodeToolsArray(t *testing.T) {
	tools := []Tool{{
		Type: "function",
		Function: ToolFunction{
			Name:        "calc_add",
			Description: "Add two ints",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"a": map[string]any{"type": "integer"},
					"b": map[string]any{"type": "integer"},
				},
			},
		},
	}}
	got, err := encodeTools(tools)
	if err != nil {
		t.Fatalf("encodeTools: %v", err)
	}
	if !strings.HasPrefix(got, "[") || !strings.HasSuffix(got, "]") {
		t.Errorf("encodeTools should produce a JSON array: %q", got)
	}
	if !strings.Contains(got, `"calc_add"`) {
		t.Errorf("expected function name in encoded tools: %q", got)
	}
}
