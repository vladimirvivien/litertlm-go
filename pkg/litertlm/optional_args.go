package litertlm

import (
	"fmt"
	"runtime"
	"unsafe"
)

// NewOptionalArgs allocates a new OptionalArgs handle. The caller owns
// the handle and must call Delete when done. Passing OptionalArgs(0)
// to Conversation.SendMessage / SendMessageStream uses the C-side defaults.
func NewOptionalArgs() (OptionalArgs, error) {
	return callStatusHandle[OptionalArgs](conversationOptionalArgsCreateFunc,
		"conversation_optional_args_create")
}

// Delete releases an OptionalArgs handle. Safe to call on the zero value (no-op).
func (o OptionalArgs) Delete() { deleteHandle(o, conversationOptionalArgsDeleteFunc) }

// SetVisualTokenBudget sets the maximum number of vision tokens any
// turn passed this OptionalArgs may consume. Effective on Gemma 4
// vision-enabled models; ignored on text-only or audio turns.
func (o OptionalArgs) SetVisualTokenBudget(n int) error {
	return setHandleScalar(o, "optional args", conversationOptionalArgsSetVisualTokenBudgetFunc, "conversation_optional_args_set_visual_token_budget", int32(n))
}

// SetMaxOutputTokens caps the number of output tokens produced per turn.
func (o OptionalArgs) SetMaxOutputTokens(n int) error {
	return setHandleScalar(o, "optional args", conversationOptionalArgsSetMaxOutputTokensFunc, "conversation_optional_args_set_max_output_tokens", int32(n))
}

// SetRepetitionPenalty configures per-turn repetition, presence, and frequency
// penalties. Passing nil clears any previously configured penalty.
func (o OptionalArgs) SetRepetitionPenalty(cfg *RepetitionPenaltyConfig) error {
	return setOptionalConfig(o, "conversation_optional_args_set_repetition_penalty_config",
		conversationOptionalArgsSetRepetitionPenaltyConfigFunc, cfg,
		newRepetitionPenaltyHandle, func(h repetitionPenaltyConfigHandle) { h.delete() })
}

// SetNoRepeatNgram configures per-turn n-gram repetition banning. Passing nil
// clears any previously configured ban.
func (o OptionalArgs) SetNoRepeatNgram(cfg *NoRepeatNgramConfig) error {
	return setOptionalConfig(o, "conversation_optional_args_set_no_repeat_ngram_config",
		conversationOptionalArgsSetNoRepeatNgramConfigFunc, cfg,
		newNoRepeatNgramHandle, func(h noRepeatNgramConfigHandle) { h.delete() })
}

// SetSuppressTokens configures per-turn token ID suppression. Passing nil
// clears any previously suppressed token list.
func (o OptionalArgs) SetSuppressTokens(cfg *SuppressTokensConfig) error {
	return setOptionalConfig(o, "conversation_optional_args_set_suppress_tokens_config",
		conversationOptionalArgsSetSuppressTokensConfigFunc, cfg,
		newSuppressTokensHandle, func(h suppressTokensConfigHandle) { h.delete() })
}

// SetThinkingConfig configures per-turn thinking/reasoning behavior. Passing
// nil clears any previously set per-turn override.
func (o OptionalArgs) SetThinkingConfig(cfg *ThinkingConfig) error {
	return setOptionalConfig(o, "conversation_optional_args_set_thinking_config",
		conversationOptionalArgsSetThinkingConfigFunc, cfg,
		newThinkingConfigHandle, func(h thinkingConfigHandle) { h.delete() })
}

// SetConstraint sets a per-turn grammar constraint (regex or JSON schema).
func (o OptionalArgs) SetConstraint(typ ConstraintType, pattern string) error {
	if o == 0 {
		return fmt.Errorf("litertlm: conversation_optional_args_set_constraint: invalid optional args")
	}
	patPtr, err := bytePtrOrNil(pattern)
	if err != nil {
		return err
	}
	ct := int32(typ)
	err = callStatus(conversationOptionalArgsSetConstraintFunc, "conversation_optional_args_set_constraint",
		unsafe.Pointer(&o), unsafe.Pointer(&ct), unsafe.Pointer(&patPtr))
	runtime.KeepAlive(patPtr)
	return err
}

// SetToolChoice sets the OpenAI-style tool_choice JSON for this turn
// (requires LiteRT-LM C API 1.1.0+). An empty string clears any prior choice.
func (o OptionalArgs) SetToolChoice(toolChoiceJSON string) error {
	if err := requireAPIVersion(1, 1); err != nil {
		return err
	}
	if o == 0 {
		return fmt.Errorf("litertlm: conversation_optional_args_set_tool_choice: invalid optional args")
	}
	ptr, err := bytePtrOrNil(toolChoiceJSON)
	if err != nil {
		return err
	}
	err = callStatus(conversationOptionalArgsSetToolChoiceFunc, "conversation_optional_args_set_tool_choice",
		unsafe.Pointer(&o), unsafe.Pointer(&ptr))
	runtime.KeepAlive(ptr)
	return err
}
