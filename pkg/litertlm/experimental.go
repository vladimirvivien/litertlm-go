package litertlm

import (
	"fmt"
	"unsafe"
)

// SetMetalResidency dynamically updates Metal residency set behavior on Apple
// GPU backends at runtime. Ignored safely on non-Apple platforms or non-GPU
// backends.
func (e Engine) SetMetalResidency(enable bool) error {
	return setHandleBool(e, "engine", experimentalEngineUpdateGpuEnableMetalResidencySetFunc,
		"experimental_engine_update_gpu_enable_metal_residency_set", enable)
}

// SetMetalResidency dynamically toggles Apple Metal residency set usage for the
// Client's GPU engine at runtime. Safely ignored on non-Apple platforms or
// non-GPU backends.
func (c *Client) SetMetalResidency(enable bool) error {
	if c == nil {
		return fmt.Errorf("litertlm: Client is nil")
	}
	engine, err := c.openEngine()
	if err != nil {
		return err
	}
	return engine.SetMetalResidency(enable)
}

// SetEnableDebugger toggles the AI Edge Debugger tracing backend for engines
// constructed with these settings (requires LiteRT-LM C API 1.1.0+ and a
// writable disk cache directory via WithCacheDir).
//
// Captures contain decoded text and intermediate activations stored unencrypted
// under `<cache_dir>/litert_lm_debugger/<session_id>/` and must be removed by
// the caller when no longer needed.
func (s EngineSettings) SetEnableDebugger(enable bool) error {
	if s == 0 {
		return fmt.Errorf("litertlm: experimental_engine_settings_set_enable_debugger: invalid engine settings")
	}
	if err := requireAPIVersion(1, 1); err != nil {
		return err
	}
	return setHandleBool(s, "engine settings", experimentalEngineSettingsSetEnableDebuggerFunc,
		"experimental_engine_settings_set_enable_debugger", enable)
}

// Delete releases a SessionDebugInfo handle.
func (d SessionDebugInfo) Delete() { deleteHandle(d, experimentalSessionDebugInfoDeleteFunc) }

// CaptureDir returns the relative debug capture directory path (for example
// "litert_lm_debugger/0") copied into Go memory.
func (d SessionDebugInfo) CaptureDir() (string, error) {
	return getHandleStringErr(d, "session debug info",
		experimentalSessionDebugInfoGetCaptureDirFunc,
		"experimental_session_debug_info_get_capture_dir")
}

// DebugInfo returns debug metadata for this Session, or (0, nil) when debugging
// is disabled or unsupported. When non-zero, caller must Delete() it.
func (s Session) DebugInfo() (SessionDebugInfo, error) {
	if s == 0 {
		return 0, fmt.Errorf("litertlm: experimental_session_get_debug_info: invalid session")
	}
	return callStatusHandle[SessionDebugInfo](experimentalSessionGetDebugInfoFunc,
		"experimental_session_get_debug_info", unsafe.Pointer(&s))
}

// SessionDebugInfo returns debug metadata for this Conversation's underlying
// session, or (0, nil) when debugging is disabled or unsupported. When
// non-zero, caller must Delete() it.
func (c Conversation) SessionDebugInfo() (SessionDebugInfo, error) {
	if c == 0 {
		return 0, fmt.Errorf("litertlm: experimental_conversation_get_session_debug_info: invalid conversation")
	}
	return callStatusHandle[SessionDebugInfo](experimentalConversationGetSessionDebugInfoFunc,
		"experimental_conversation_get_session_debug_info", unsafe.Pointer(&c))
}

// DebugCaptureDir returns the relative debug capture directory for this Session,
// or an empty string when debugging is disabled or unsupported.
func (s Session) DebugCaptureDir() (string, error) {
	info, err := s.DebugInfo()
	if err != nil || info == 0 {
		return "", err
	}
	defer info.Delete()
	return info.CaptureDir()
}

// DebugCaptureDir returns the relative debug capture directory for this
// Conversation's underlying session, or an empty string when debugging is
// disabled or unsupported.
func (c Conversation) DebugCaptureDir() (string, error) {
	info, err := c.SessionDebugInfo()
	if err != nil || info == 0 {
		return "", err
	}
	defer info.Delete()
	return info.CaptureDir()
}

// DebugCaptureDir returns the relative AI Edge Debugger capture directory for
// this Chat's underlying session (empty string when debugging is disabled or
// unsupported).
func (ch *Chat) DebugCaptureDir() (string, error) {
	if ch == nil {
		return "", fmt.Errorf("litertlm: Chat is nil")
	}
	conv, err := ch.openConv()
	if err != nil {
		return "", err
	}
	return conv.DebugCaptureDir()
}
