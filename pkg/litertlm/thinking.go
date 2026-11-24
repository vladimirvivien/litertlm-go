package litertlm

import (
	"encoding/json"
	"fmt"
	"slices"
)

// ThinkingConfig configures model reasoning/thinking generation. Zero
// TokenBudget preserves the C-side unlimited (-1) budget default; pass -1 or a
// positive integer to set the reasoning token limit explicitly.
type ThinkingConfig struct {
	Enabled     bool
	TokenBudget int
}

type thinkingConfigHandle uintptr

func (h thinkingConfigHandle) delete() { deleteHandle(h, thinkingConfigDeleteFunc) }

func newThinkingConfigHandle(cfg ThinkingConfig) (thinkingConfigHandle, error) {
	h, err := callStatusHandle[thinkingConfigHandle](
		thinkingConfigCreateFunc,
		"thinking_config_create",
	)
	if err != nil {
		return 0, err
	}
	if err = setHandleBool(h, "thinking config",
		thinkingConfigSetEnableThinkingFunc,
		"thinking_config_set_enable_thinking", cfg.Enabled); err != nil {
		h.delete()
		return 0, err
	}
	if cfg.TokenBudget != 0 {
		if err = setHandleScalar(h, "thinking config",
			thinkingConfigSetThinkingTokenBudgetFunc,
			"thinking_config_set_thinking_token_budget", int32(cfg.TokenBudget)); err != nil {
			h.delete()
			return 0, err
		}
	}
	return h, nil
}

// ToolChoice specifies per-turn OpenAI-compatible tool selection policy for
// constrained decoding (requires LiteRT-LM C API 1.1.0+).
type ToolChoice struct {
	mode         string
	functionName string
	allowedMode  string
	allowedNames []string
	rawJSON      string
	set          bool
}

// ToolChoiceAuto lets the model decide whether to call a tool or reply directly.
func ToolChoiceAuto() ToolChoice { return ToolChoice{mode: "auto", set: true} }

// ToolChoiceNone prevents tool calls on the turn.
func ToolChoiceNone() ToolChoice { return ToolChoice{mode: "none", set: true} }

// ToolChoiceRequired forces at least one tool call from the registered tools.
func ToolChoiceRequired() ToolChoice { return ToolChoice{mode: "required", set: true} }

// ToolChoiceFunction forces invocation of the named function.
func ToolChoiceFunction(name string) ToolChoice {
	return ToolChoice{functionName: name, set: true}
}

// ToolChoiceAllowedTools restricts callable tools to names using mode ("auto" or "required").
func ToolChoiceAllowedTools(mode string, names ...string) ToolChoice {
	return ToolChoice{
		allowedMode:  mode,
		allowedNames: slices.Clone(names),
		set:          true,
	}
}

// ToolChoiceRawJSON passes an explicit OpenAI tool_choice JSON value verbatim.
func ToolChoiceRawJSON(rawJSON string) ToolChoice {
	return ToolChoice{rawJSON: rawJSON, set: true}
}

// IsZero reports whether t represents an unset ToolChoice.
func (t ToolChoice) IsZero() bool { return !t.set }

// JSON encodes t as an OpenAI tool_choice JSON string expected by the C API.
func (t ToolChoice) JSON() (string, error) {
	if !t.set {
		return "", nil
	}
	if t.rawJSON != "" {
		return t.rawJSON, nil
	}
	switch {
	case t.mode != "":
		b, err := json.Marshal(t.mode)
		if err != nil {
			return "", fmt.Errorf("marshal tool choice mode: %w", err)
		}
		return string(b), nil
	case t.functionName != "":
		b, err := json.Marshal(map[string]any{
			"type":     "function",
			"function": map[string]string{"name": t.functionName},
		})
		if err != nil {
			return "", fmt.Errorf("marshal tool choice function: %w", err)
		}
		return string(b), nil
	case len(t.allowedNames) > 0 || t.allowedMode != "":
		mode := t.allowedMode
		if mode == "" {
			mode = "auto"
		}
		if len(t.allowedNames) == 0 {
			return "", fmt.Errorf("litertlm: tool choice allowed_tools requires at least one tool name")
		}
		tools := make([]map[string]any, len(t.allowedNames))
		for i, name := range t.allowedNames {
			tools[i] = map[string]any{
				"type":     "function",
				"function": map[string]string{"name": name},
			}
		}
		b, err := json.Marshal(map[string]any{
			"type": "allowed_tools",
			"allowed_tools": map[string]any{
				"mode":  mode,
				"tools": tools,
			},
		})
		if err != nil {
			return "", fmt.Errorf("marshal tool choice allowed_tools: %w", err)
		}
		return string(b), nil
	default:
		return "", fmt.Errorf("litertlm: empty tool choice")
	}
}
