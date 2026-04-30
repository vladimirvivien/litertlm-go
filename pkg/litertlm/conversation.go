package litertlm

import (
	"errors"
	"unsafe"

	"github.com/vladimirvivien/litertlm-go/pkg/utils"
)

// NewConversationConfig builds a ConversationConfig with the provided
// metadata. Any of systemMessageJSON, toolsJSON, and messagesJSON may be
// empty strings to omit them. sessionConfig may be 0 to let the C API
// pick defaults.
//
// The C API exposes config construction as `create()` returning an empty
// handle, plus a setter per field (see c/engine.h: set_session_config,
// set_system_message, set_tools, set_messages,
// set_enable_constrained_decoding). The setters copy their string
// arguments into std::string members of the underlying struct, so the
// Go-side bytes do not need to outlive the call. The `engine` parameter
// is retained for callsite symmetry with NewConversation but is not
// actually consumed by the C config constructor.
func NewConversationConfig(
	engine Engine,
	sessionConfig SessionConfig,
	systemMessageJSON string,
	toolsJSON string,
	messagesJSON string,
	enableConstrainedDecoding bool,
) (ConversationConfig, error) {
	if engine == 0 {
		return 0, errors.New("litertlm: conversation_config_create: invalid engine")
	}

	var c ConversationConfig
	conversationConfigCreateFunc.Call(unsafe.Pointer(&c))
	if c == 0 {
		return 0, errors.New("litertlm: conversation_config_create failed")
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
		var enable uint8 = 1
		conversationConfigSetEnableConstrainedDecodingFunc.Call(
			nil,
			unsafe.Pointer(&c),
			unsafe.Pointer(&enable),
		)
	}

	return c, nil
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

// SendMessage runs a blocking multi-turn send against the conversation and
// returns the JSON response (copied into Go memory).
func (c Conversation) SendMessage(messageJSON, extraContext string) (string, error) {
	if c == 0 {
		return "", errors.New("litertlm: send_message: invalid conversation")
	}
	msgPtr, err := utils.BytePtrFromString(messageJSON)
	if err != nil {
		return "", err
	}
	ctxPtr, err := bytePtrOrNil(extraContext)
	if err != nil {
		return "", err
	}

	var handle JsonResponse
	conversationSendMessageFunc.Call(
		unsafe.Pointer(&handle),
		unsafe.Pointer(&c),
		unsafe.Pointer(&msgPtr),
		unsafe.Pointer(&ctxPtr),
	)
	if handle == 0 {
		return "", errors.New("litertlm: conversation_send_message failed")
	}
	defer handle.Delete()

	return handle.String(), nil
}

// Cancel requests cancellation of an in-flight streaming send.
func (c Conversation) Cancel() {
	if c == 0 {
		return
	}
	conversationCancelProcessFunc.Call(nil, unsafe.Pointer(&c))
}

// BenchmarkInfo returns conversation-level benchmark metrics. Requires the
// engine to have been created with EnableBenchmark(). Caller must Delete().
func (c Conversation) BenchmarkInfo() (BenchmarkInfo, error) {
	if c == 0 {
		return 0, errors.New("litertlm: benchmark_info: invalid conversation")
	}
	var b BenchmarkInfo
	conversationGetBenchmarkInfoFunc.Call(unsafe.Pointer(&b), unsafe.Pointer(&c))
	if b == 0 {
		return 0, errors.New("litertlm: conversation_get_benchmark_info failed")
	}
	return b, nil
}

// ---- JsonResponse --------------------------------------------------------

// Delete releases a JsonResponse handle.
func (j JsonResponse) Delete() {
	if j == 0 {
		return
	}
	jsonResponseDeleteFunc.Call(nil, unsafe.Pointer(&j))
}

// String returns the JSON payload as a Go string, copied into Go memory.
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

// ---- helpers -------------------------------------------------------------

// bytePtrOrNil returns a null-terminated byte pointer for s, or nil if s is
// empty. Several C entry points accept NULL for "unset".
func bytePtrOrNil(s string) (*byte, error) {
	if s == "" {
		return nil, nil
	}
	return utils.BytePtrFromString(s)
}
