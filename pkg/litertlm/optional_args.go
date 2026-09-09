package litertlm

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
