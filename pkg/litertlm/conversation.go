package litertlm

import (
	"fmt"
	"unsafe"

	"github.com/vladimirvivien/litertlm-go/pkg/utils"
)

// NewConversationConfig builds a ConversationConfig populated with the
// given fields. Empty JSON strings or zero sessionConfig are skipped.
//
// systemMessageJSON should be the content (a JSON string or content
// array), not a `{role,content}` envelope — the C side wraps it itself
// (see c/engine.cc litert_lm_conversation_create). The C config
// constructor ignores its `engine` argument; the wrapper still
// requires it as a non-zero handle for callsite symmetry with
// NewConversation. Setters copy strings into std::string fields, so
// the Go bytes need not outlive the call.
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

	c, err := callForHandle[ConversationConfig](conversationConfigCreateFunc,
		"conversation_config_create")
	if err != nil {
		return 0, err
	}

	if sessionConfig != 0 {
		conversationConfigSetSessionConfigFunc.Call(
			nil,
			unsafe.Pointer(&c),
			unsafe.Pointer(&sessionConfig),
		)
	}

	if systemMessageJSON != "" {
		sysPtr, err := utils.BytePtrFromString(systemMessageJSON)
		if err != nil {
			c.Delete()
			return 0, err
		}
		conversationConfigSetSystemMessageFunc.Call(
			nil,
			unsafe.Pointer(&c),
			unsafe.Pointer(&sysPtr),
		)
	}

	if toolsJSON != "" {
		toolsPtr, err := utils.BytePtrFromString(toolsJSON)
		if err != nil {
			c.Delete()
			return 0, err
		}
		conversationConfigSetToolsFunc.Call(
			nil,
			unsafe.Pointer(&c),
			unsafe.Pointer(&toolsPtr),
		)
	}

	if messagesJSON != "" {
		msgsPtr, err := utils.BytePtrFromString(messagesJSON)
		if err != nil {
			c.Delete()
			return 0, err
		}
		conversationConfigSetMessagesFunc.Call(
			nil,
			unsafe.Pointer(&c),
			unsafe.Pointer(&msgsPtr),
		)
	}

	if enableConstrainedDecoding {
		conversationConfigSetEnableConstrainedDecodingFunc.Call(
			nil,
			unsafe.Pointer(&c),
			unsafe.Pointer(new(uint8(1))),
		)
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
	ctxPtr, err := utils.BytePtrFromString(extraContextJSON)
	if err != nil {
		return err
	}
	conversationConfigSetExtraContextFunc.Call(nil, unsafe.Pointer(&c), unsafe.Pointer(&ctxPtr))
	return nil
}

// SetFilterChannelContentFromKVCache toggles whether reasoning-channel
// content is excluded from the conversation's KV cache. When on, the
// model's internal-reasoning tokens (between <|channel> ... <channel|>
// markers) are not persisted across turns.
func (c ConversationConfig) SetFilterChannelContentFromKVCache(on bool) {
	if c == 0 {
		return
	}
	var v uint8
	if on {
		v = 1
	}
	conversationConfigSetFilterChannelContentFromKVCacheFunc.Call(
		nil, unsafe.Pointer(&c), unsafe.Pointer(&v),
	)
}

// Delete releases a ConversationConfig handle.
func (c ConversationConfig) Delete() {
	if c == 0 {
		return
	}
	conversationConfigDeleteFunc.Call(nil, unsafe.Pointer(&c))
}

// Delete releases a Conversation handle.
func (c Conversation) Delete() {
	if c == 0 {
		return
	}
	conversationDeleteFunc.Call(nil, unsafe.Pointer(&c))
}

// SendMessage runs a blocking multi-turn send and returns the JSON
// response (copied into Go memory). opts carries per-call knobs such
// as the visual token budget; pass OptionalArgs(0) for C-side
// defaults.
func (c Conversation) SendMessage(messageJSON, extraContext string, opts OptionalArgs) (string, error) {
	if c == 0 {
		return "", fmt.Errorf("litertlm: send_message: invalid conversation")
	}
	msgPtr, err := utils.BytePtrFromString(messageJSON)
	if err != nil {
		return "", err
	}
	ctxPtr, err := bytePtrOrNil(extraContext)
	if err != nil {
		return "", err
	}

	handle, err := callForHandle[JsonResponse](conversationSendMessageFunc,
		"conversation_send_message",
		unsafe.Pointer(&c),
		unsafe.Pointer(&msgPtr),
		unsafe.Pointer(&ctxPtr),
		unsafe.Pointer(&opts),
	)
	if err != nil {
		return "", err
	}
	defer handle.Delete()
	return handle.String(), nil
}

// RenderMessage runs messageJSON through the chat template and
// returns the rendered string. Useful for debugging template output
// or for inspecting how a turn will appear to the model. The C side
// owns the returned buffer; this method copies into Go memory before
// returning so the result remains valid for the caller's lifetime.
func (c Conversation) RenderMessage(messageJSON string) (string, error) {
	if c == 0 {
		return "", fmt.Errorf("litertlm: render_message: invalid conversation")
	}
	msgPtr, err := utils.BytePtrFromString(messageJSON)
	if err != nil {
		return "", err
	}
	var rawPtr *byte
	conversationRenderMessageToStringFunc.Call(
		unsafe.Pointer(&rawPtr),
		unsafe.Pointer(&c),
		unsafe.Pointer(&msgPtr),
	)
	if rawPtr == nil {
		return "", fmt.Errorf("litertlm: render_message: C side returned NULL")
	}
	return utils.BytePtrToString(rawPtr), nil
}

// Cancel requests cancellation of an in-flight streaming send.
func (c Conversation) Cancel() {
	if c == 0 {
		return
	}
	conversationCancelProcessFunc.Call(nil, unsafe.Pointer(&c))
}

// BenchmarkInfo returns conversation-level benchmark metrics. Requires
// EngineSettings.EnableBenchmark() at engine construction time.
func (c Conversation) BenchmarkInfo() (BenchmarkInfo, error) {
	if c == 0 {
		return 0, fmt.Errorf("litertlm: benchmark_info: invalid conversation")
	}
	return callForHandle[BenchmarkInfo](conversationGetBenchmarkInfoFunc,
		"conversation_get_benchmark_info", unsafe.Pointer(&c))
}

// TokenCount returns the number of tokens currently held in the
// conversation's KV cache (prefill + decode). Unlike the BenchmarkInfo
// counts, it does not require EnableBenchmark.
//
// Requires LiteRT-LM v0.13.1 or newer.
func (c Conversation) TokenCount() (int, error) {
	if c == 0 {
		return 0, fmt.Errorf("litertlm: conversation_get_token_count: invalid conversation")
	}
	var v int64
	conversationGetTokenCountFunc.Call(unsafe.Pointer(&v), unsafe.Pointer(&c))
	v = int64(int32(v))
	if v < 0 {
		return 0, fmt.Errorf("litertlm: conversation_get_token_count failed (code=%d)", v)
	}
	return int(v), nil
}

// ---- JsonResponse --------------------------------------------------------

// Delete releases a JsonResponse handle.
func (j JsonResponse) Delete() {
	if j == 0 {
		return
	}
	jsonResponseDeleteFunc.Call(nil, unsafe.Pointer(&j))
}

// String returns the JSON payload as a Go string (copied into Go memory).
func (j JsonResponse) String() string {
	if j == 0 {
		return ""
	}
	var ptr *byte
	jsonResponseGetStringFunc.Call(unsafe.Pointer(&ptr), unsafe.Pointer(&j))
	if ptr == nil {
		return ""
	}
	return utils.BytePtrToString(ptr)
}
