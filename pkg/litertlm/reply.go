package litertlm

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Reply is one assistant turn returned by Chat.Send / Chat.SendToolResult.
// It wraps the JSON document the C side surfaces (the chat template's
// rendering of the model's output, parsed into structured fields).
//
// The underlying C-side JSON handle is freed during the SendMessage
// call that produced the Reply, so a Reply has no external lifetime
// concerns and is safe to pass around or hold indefinitely.
type Reply struct {
	raw       string
	role      string
	content   []replyContentPart
	toolCalls []ToolCall
}

type replyContentPart struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// Text returns the concatenation of every text content part. Empty if
// the assistant produced no text (e.g. the turn is purely a tool_call).
func (r *Reply) Text() string {
	if r == nil {
		return ""
	}
	var b strings.Builder
	for _, p := range r.content {
		if p.Type == "text" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// ToolCalls returns the function-call requests the model made on this
// turn. Empty when the model answered with text only.
func (r *Reply) ToolCalls() []ToolCall {
	if r == nil {
		return nil
	}
	return r.toolCalls
}

// HasToolCalls reports whether this reply requests one or more tool
// invocations.
func (r *Reply) HasToolCalls() bool {
	return r != nil && len(r.toolCalls) > 0
}

// Raw returns the original JSON document as-returned by the C API.
// Useful for logging / debugging when fields the high-level Reply
// doesn't surface (image parts, audio parts, reasoning channels) are
// present.
func (r *Reply) Raw() string {
	if r == nil {
		return ""
	}
	return r.raw
}

// parseReply unmarshals the C-side JSON envelope into a *Reply and
// strips Gemma 4 quote markers (`<|"|>`) from any string-typed tool
// arguments. The marker leakage is a known C-side passthrough quirk
// for string-typed function args; we normalise once here so callers
// always see clean values.
func parseReply(raw string) (*Reply, error) {
	var msg struct {
		Role      string             `json:"role"`
		Content   []replyContentPart `json:"content,omitempty"`
		ToolCalls []ToolCall         `json:"tool_calls,omitempty"`
	}
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		return nil, fmt.Errorf("litertlm: parse reply: %w (raw=%s)", err, raw)
	}

	for i := range msg.ToolCalls {
		args := msg.ToolCalls[i].Function.Arguments
		for k, v := range args {
			if s, ok := v.(string); ok {
				args[k] = strings.ReplaceAll(s, `<|"|>`, "")
			}
		}
	}

	return &Reply{
		raw:       raw,
		role:      msg.Role,
		content:   msg.Content,
		toolCalls: msg.ToolCalls,
	}, nil
}
