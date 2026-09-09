package litertlm

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/vladimirvivien/litertlm-go/internal/cstr"
)

// NewConversationConfig builds a ConversationConfig populated with the
// given fields. Empty JSON strings or zero sessionConfig are skipped.
//
// systemMessageJSON should be the content (a JSON string or content
// array), not a `{role,content}` envelope — the C side wraps it itself.
func NewConversationConfig(
	engine Engine,
	sessionConfig SessionConfig,
	systemMessageJSON string,
	toolsJSON string,
	messagesJSON string,
	enableConstrainedDecoding bool,
) (ConversationConfig, error) {
	if engine == 0 {
		return 0, fmt.Errorf("litertlm: conversation_config_create: invalid engine")
	}

	c, err := callStatusHandle[ConversationConfig](conversationConfigCreateFunc, "conversation_config_create")
	if err != nil {
		return 0, err
	}

	if sessionConfig != 0 {
		if err = callStatus(
			conversationConfigSetSessionConfigFunc,
			"conversation_config_set_session_config",
			unsafe.Pointer(&c),
			unsafe.Pointer(&sessionConfig),
		); err != nil {
			c.Delete()
			return 0, err
		}
	}

	for _, step := range []struct {
		fn  *lazyFun
		op  string
		val string
	}{
		{conversationConfigSetSystemMessageFunc, "conversation_config_set_system_message", systemMessageJSON},
		{conversationConfigSetToolsFunc, "conversation_config_set_tools", toolsJSON},
		{conversationConfigSetMessagesFunc, "conversation_config_set_messages", messagesJSON},
	} {
		if step.val == "" {
			continue
		}
		if err = setHandleString(c, "conversation config", step.fn, step.op, step.val); err != nil {
			c.Delete()
			return 0, err
		}
	}

	if enableConstrainedDecoding {
		if err = setHandleBool(c, "conversation config", conversationConfigSetEnableConstrainedDecodingFunc, "conversation_config_set_enable_constrained_decoding", true); err != nil {
			c.Delete()
			return 0, err
		}
	}

	return c, nil
}

// SetExtraContext attaches an extra-context JSON string used in the
// conversation preface. Empty string is a no-op. C-side copies the
// bytes; the Go buffer need not outlive the call.
func (c ConversationConfig) SetExtraContext(extraContextJSON string) error {
	if c == 0 || extraContextJSON == "" {
		return nil
	}
	return setHandleString(c, "conversation config", conversationConfigSetExtraContextFunc, "conversation_config_set_extra_context", extraContextJSON)
}

// SetFilterChannelContentFromKVCache toggles whether reasoning-channel
// content is excluded from the conversation's KV cache. When on, the
// model's internal-reasoning tokens (between <|channel> ... <channel|>)
// are not persisted across turns.
func (c ConversationConfig) SetFilterChannelContentFromKVCache(on bool) error {
	return setHandleBool(c, "conversation config", conversationConfigSetFilterChannelContentFromKVCacheFunc, "conversation_config_set_filter_channel_content_from_kv_cache", on)
}

// SetStreamToolCalls configures whether tool call tokens should be emitted to
// streaming callers rather than held until complete, and under what channel name.
func (c ConversationConfig) SetStreamToolCalls(stream bool, channelName string) error {
	if c == 0 {
		return fmt.Errorf("litertlm: set_stream_tool_calls: invalid conversation config")
	}
	var v uint8
	if stream {
		v = 1
	}
	chPtr, err := bytePtrOrNil(channelName)
	if err != nil {
		return err
	}
	err = callStatus(
		conversationConfigSetStreamToolCallsFunc,
		"conversation_config_set_stream_tool_calls",
		unsafe.Pointer(&c), unsafe.Pointer(&v), unsafe.Pointer(&chPtr),
	)
	runtime.KeepAlive(chPtr)
	return err
}

// Delete releases a ConversationConfig handle.
func (c ConversationConfig) Delete() { deleteHandle(c, conversationConfigDeleteFunc) }

// Delete releases a Conversation handle.
func (c Conversation) Delete() { deleteHandle(c, conversationDeleteFunc) }

// Clone returns a new Conversation that mirrors c's prefilled state —
// activation frames and KV cache included. O(1) compared to opening a
// new Conversation and re-prefilling the same prompt.
func (c Conversation) Clone() (Conversation, error) {
	if c == 0 {
		return 0, fmt.Errorf("litertlm: conversation_clone: invalid conversation")
	}
	return callStatusHandle[Conversation](conversationCloneFunc, "conversation_clone", unsafe.Pointer(&c))
}

// SendMessage runs a blocking multi-turn send and returns the JSON
// response (copied into Go memory). opts carries per-call knobs such
// as the visual token budget; pass OptionalArgs(0) for C-side defaults.
func (c Conversation) SendMessage(messageJSON, extraContext string, opts OptionalArgs) (string, error) {
	if c == 0 {
		return "", fmt.Errorf("litertlm: send_message: invalid conversation")
	}
	msgPtr, err := cstr.BytePtrFromString(messageJSON)
	if err != nil {
		return "", err
	}
	ctxPtr, err := bytePtrOrNil(extraContext)
	if err != nil {
		return "", err
	}

	handle, err := callStatusHandle[JSONResponse](conversationSendMessageFunc,
		"conversation_send_message",
		unsafe.Pointer(&c),
		unsafe.Pointer(&msgPtr),
		unsafe.Pointer(&ctxPtr),
		unsafe.Pointer(&opts),
	)
	runtime.KeepAlive(msgPtr)
	runtime.KeepAlive(ctxPtr)
	if err != nil {
		return "", err
	}
	defer handle.Delete()
	return handle.String(), nil
}

// RenderMessage runs messageJSON through the chat template and
// returns the rendered string.
func (c Conversation) RenderMessage(messageJSON string) (string, error) {
	if c == 0 {
		return "", fmt.Errorf("litertlm: render_message: invalid conversation")
	}
	msgPtr, err := cstr.BytePtrFromString(messageJSON)
	if err != nil {
		return "", err
	}
	s, ok, err := callStatusString(
		conversationRenderMessageToStringFunc,
		"conversation_render_message_to_string",
		unsafe.Pointer(&c),
		unsafe.Pointer(&msgPtr),
	)
	runtime.KeepAlive(msgPtr)
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("litertlm: render_message: C side returned NULL")
	}
	return s, nil
}

// RenderPreface renders the conversation's preface according to the template.
func (c Conversation) RenderPreface() (string, error) {
	if c == 0 {
		return "", fmt.Errorf("litertlm: render_preface: invalid conversation")
	}
	s, ok, err := callStatusString(
		conversationRenderPrefaceToStringFunc,
		"conversation_render_preface_to_string",
		unsafe.Pointer(&c),
	)
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("litertlm: render_preface: C side returned NULL")
	}
	return s, nil
}

// Cancel requests cancellation of an in-flight streaming send.
func (c Conversation) Cancel() {
	if c == 0 {
		return
	}
	var status int32
	_ = conversationCancelProcessFunc.callErr(unsafe.Pointer(&status), unsafe.Pointer(&c))
}

// BenchmarkInfo returns conversation-level benchmark metrics. Requires
// EngineSettings.EnableBenchmark() at engine construction time.
func (c Conversation) BenchmarkInfo() (BenchmarkInfo, error) {
	if c == 0 {
		return 0, fmt.Errorf("litertlm: benchmark_info: invalid conversation")
	}
	return callStatusHandle[BenchmarkInfo](conversationGetBenchmarkInfoFunc,
		"conversation_get_benchmark_info", unsafe.Pointer(&c))
}

// TokenCount returns the number of tokens currently held in the
// conversation's KV cache (prefill + decode). Unlike the BenchmarkInfo
// counts, it does not require EnableBenchmark.
func (c Conversation) TokenCount() (int, error) {
	if c == 0 {
		return 0, fmt.Errorf("litertlm: conversation_get_token_count: invalid conversation")
	}
	n, err := callStatusScalar[uint64](conversationGetTokenCountFunc,
		"conversation_get_token_count", unsafe.Pointer(&c))
	if err != nil {
		return 0, err
	}
	return int(n), nil
}

// ---- JSONResponse --------------------------------------------------------

// Delete releases a JSONResponse handle.
func (j JSONResponse) Delete() { deleteHandle(j, jsonResponseDeleteFunc) }

// String returns the JSON payload as a Go string (copied into Go memory).
func (j JSONResponse) String() string {
	if j == 0 {
		return ""
	}
	s, ok, err := callStatusString(jsonResponseGetStringFunc, "json_response_get_string", unsafe.Pointer(&j))
	if err != nil || !ok {
		return ""
	}
	return s
}
