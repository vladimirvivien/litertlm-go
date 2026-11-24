package litertlm

import (
	"encoding/json"
	"fmt"
	"strings"
)

type toolResult struct {
	name     string
	response any
}

func concatTextParts(parts []replyContentPart) (string, bool) {
	if len(parts) == 1 && parts[0].Type == "text" {
		return parts[0].Text, true
	}
	var b strings.Builder
	found := false
	for _, p := range parts {
		if p.Type == "text" {
			found = true
			b.WriteString(p.Text)
		}
	}
	return b.String(), found
}

func stripToolQuoteMarkers(calls []ToolCall) {
	for i := range calls {
		args := calls[i].Function.Arguments
		for k, v := range args {
			if s, ok := v.(string); ok {
				args[k] = strings.ReplaceAll(s, `<|"|>`, "")
			}
		}
	}
}

func extractStreamChunkText(raw string) string {
	if raw == "" {
		return ""
	}
	var msg struct {
		Content []replyContentPart `json:"content"`
	}
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		return raw
	}
	text, _ := concatTextParts(msg.Content)
	return text
}

func extractStreamEnvelope(raw string) (string, []ToolCall) {
	if raw == "" {
		return "", nil
	}
	var msg struct {
		Content   []replyContentPart `json:"content,omitempty"`
		ToolCalls []ToolCall         `json:"tool_calls,omitempty"`
	}
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		return raw, nil
	}
	stripToolQuoteMarkers(msg.ToolCalls)
	text, _ := concatTextParts(msg.Content)
	return text, msg.ToolCalls
}

func buildOptionalArgs(cfg runtimeConfig) (OptionalArgs, error) {
	hasConstraint := cfg.constraintType != ConstraintNone || cfg.constraintPattern != ""
	hasToolChoice := !cfg.toolChoice.IsZero()
	if cfg.visualTokenBudget == nil && cfg.maxOutputTokens <= 0 &&
		cfg.thinking == nil && !hasConstraint && !hasToolChoice &&
		cfg.repetitionPenalty == nil && cfg.noRepeatNgram == nil && cfg.suppressTokens == nil {
		return 0, nil
	}
	o, err := NewOptionalArgs()
	if err != nil {
		return 0, fmt.Errorf("litertlm: per-call options: %w", err)
	}
	fail := func(err error) (OptionalArgs, error) {
		o.Delete()
		return 0, fmt.Errorf("litertlm: per-call options: %w", err)
	}
	if cfg.visualTokenBudget != nil {
		if err = o.SetVisualTokenBudget(*cfg.visualTokenBudget); err != nil {
			return fail(err)
		}
	}
	if cfg.maxOutputTokens > 0 {
		if err = o.SetMaxOutputTokens(cfg.maxOutputTokens); err != nil {
			return fail(err)
		}
	}
	if cfg.thinking != nil {
		if err = o.SetThinkingConfig(cfg.thinking); err != nil {
			return fail(err)
		}
	}
	if hasConstraint {
		if err = o.SetConstraint(cfg.constraintType, cfg.constraintPattern); err != nil {
			return fail(err)
		}
	}
	if hasToolChoice {
		rawTC, encErr := cfg.toolChoice.JSON()
		if encErr != nil {
			return fail(encErr)
		}
		if err = o.SetToolChoice(rawTC); err != nil {
			return fail(err)
		}
	}
	if cfg.repetitionPenalty != nil {
		if err = o.SetRepetitionPenalty(cfg.repetitionPenalty); err != nil {
			return fail(err)
		}
	}
	if cfg.noRepeatNgram != nil {
		if err = o.SetNoRepeatNgram(cfg.noRepeatNgram); err != nil {
			return fail(err)
		}
	}
	if cfg.suppressTokens != nil {
		if err = o.SetSuppressTokens(cfg.suppressTokens); err != nil {
			return fail(err)
		}
	}
	return o, nil
}

func encodeSystemPrompt(cfg chatConfig) (string, error) {
	if !cfg.systemPromptSet {
		return "", nil
	}
	b, err := json.Marshal(cfg.systemPrompt)
	if err != nil {
		return "", fmt.Errorf("marshal system prompt: %w", err)
	}
	return string(b), nil
}

func encodeTools(defs []ToolDefinition) (string, error) {
	if len(defs) == 0 {
		return "", nil
	}
	list := make([]map[string]any, len(defs))
	for i, d := range defs {
		list[i] = map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        d.Name(),
				"description": d.Description(),
				"parameters":  d.Parameters(),
			},
		}
	}
	b, err := json.Marshal(list)
	if err != nil {
		return "", fmt.Errorf("marshal tools: %w", err)
	}
	return string(b), nil
}

func encodeMessages(msgs []Message) (string, error) {
	if len(msgs) == 0 {
		return "", nil
	}
	list := make([]map[string]any, len(msgs))
	for i, m := range msgs {
		if m.Role == "" {
			return "", fmt.Errorf("litertlm: initial message %d: empty role", i)
		}
		list[i] = map[string]any{
			"role":    m.Role,
			"content": renderPartsContent(m.Parts),
		}
	}
	b, err := json.Marshal(list)
	if err != nil {
		return "", fmt.Errorf("marshal initial messages: %w", err)
	}
	return string(b), nil
}

func encodeToolResults(results []toolResult) (string, error) {
	content := make([]map[string]any, len(results))
	for i, r := range results {
		content[i] = map[string]any{"name": r.name, "response": r.response}
	}
	b, err := json.Marshal(map[string]any{
		"role":    "tool",
		"content": content,
	})
	if err != nil {
		return "", fmt.Errorf("marshal tool results: %w", err)
	}
	return string(b), nil
}
