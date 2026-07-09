package litertlm

import (
	"fmt"
	"unsafe"
)

// NewOptionalArgs allocates a new OptionalArgs handle. The caller owns
// the handle and must call Delete when done. Passing a nil
// OptionalArgs to Conversation.SendMessage / SendMessageStream uses
// the C-side defaults.
func NewOptionalArgs() (OptionalArgs, error) {
	return callForHandle[OptionalArgs](conversationOptionalArgsCreateFunc,
		"conversation_optional_args_create")
}

// Delete releases an OptionalArgs handle. Safe to call on the zero
// value (no-op).
func (o OptionalArgs) Delete() {
	if o == 0 {
		return
	}
	conversationOptionalArgsDeleteFunc.Call(nil, unsafe.Pointer(&o))
}

// SetVisualTokenBudget sets the maximum number of vision tokens any
// turn passed this OptionalArgs may consume. Effective on Gemma 4
// vision-enabled models; ignored on text-only or audio turns.
func (o OptionalArgs) SetVisualTokenBudget(n int) {
	if o == 0 {
		return
	}
	conversationOptionalArgsSetVisualTokenBudgetFunc.Call(
		nil,
		unsafe.Pointer(&o),
		unsafe.Pointer(new(int32(n))),
	)
}

// SetMaxOutputTokens caps the number of output tokens produced per turn.
func (o OptionalArgs) SetMaxOutputTokens(n int) {
	if o == 0 {
		return
	}
	conversationOptionalArgsSetMaxOutputTokensFunc.Call(
		nil,
		unsafe.Pointer(&o),
		unsafe.Pointer(new(int32(n))),
	)
}

// Clone returns a new Conversation that mirrors c's prefilled state —
// activation frames and KV cache included. O(1) compared to opening a
// new Conversation and re-prefilling the same prompt.
//
// Useful for branching tool loops (try N tools off one prefilled
// prompt) and for structured-output retries. The returned handle is
// independent of c; the caller owns it and must Delete it.
//
// Clone delegates to the underlying executor's Session.Clone. As of
// LiteRT-LM v0.12.0 the LiteRT executor used by Gemma 4 on CPU and
// GPU returns Unimplemented, so Clone fails with
// `conversation_clone failed`. Only executors that implement
// Session.Clone (currently a subset of upstream backends) succeed.
func (c Conversation) Clone() (Conversation, error) {
	if c == 0 {
		return 0, fmt.Errorf("litertlm: conversation_clone: invalid conversation")
	}
	return callForHandle[Conversation](conversationCloneFunc,
		"conversation_clone",
		unsafe.Pointer(&c),
	)
}
