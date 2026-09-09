package litertlm

import (
	"encoding/json"
	"fmt"
)

// Reply is one assistant turn returned by Chat.Send / Chat.SendToolResult.
// It wraps the JSON document surfaced by the C API.
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
	text, _ := concatTextParts(r.content)
	return text
}

// ToolCalls returns a defensive copy of the function-call requests the
// model made on this turn. Empty when the model answered with text only.
func (r *Reply) ToolCalls() []ToolCall {
	if r == nil || len(r.toolCalls) == 0 {
		return nil
	}
	out := make([]ToolCall, len(r.toolCalls))
	for i, tc := range r.toolCalls {
		out[i] = ToolCall{
			Type: tc.Type,
			Function: ToolCallFunction{
				Name:      tc.Function.Name,
				Arguments: cloneJSONMap(tc.Function.Arguments),
			},
		}
	}
	return out
}

// HasToolCalls reports whether this reply requests one or more tool
// invocations.
func (r *Reply) HasToolCalls() bool {
	return r != nil && len(r.toolCalls) > 0
}

// Raw returns the original JSON document as returned by the C API.
func (r *Reply) Raw() string {
	if r == nil {
		return ""
	}
	return r.raw
}

// parseReply unmarshals the C-side JSON envelope into a *Reply and
// strips Gemma 4 quote markers (`<|"|>`) from any string-typed tool arguments.
func parseReply(raw string) (*Reply, error) {
	var msg struct {
		Role      string             `json:"role"`
		Content   []replyContentPart `json:"content,omitempty"`
		ToolCalls []ToolCall         `json:"tool_calls,omitempty"`
	}
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		return nil, fmt.Errorf("litertlm: parse reply: %w (raw=%s)", err, raw)
	}

	stripToolQuoteMarkers(msg.ToolCalls)

	return &Reply{
		raw:       raw,
		role:      msg.Role,
		content:   msg.Content,
		toolCalls: msg.ToolCalls,
	}, nil
}
