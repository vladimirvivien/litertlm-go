package litertlm

import (
	"fmt"
	"unsafe"

	"github.com/vladimirvivien/litertlm-go/pkg/utils"
)

// NewEngineSettings constructs an EngineSettings handle. vision and audio
// may be nil to leave those backends unset. Caller must Delete().
func NewEngineSettings(modelPath, backend string, vision, audio *string) (EngineSettings, error) {
	pathPtr, err := utils.BytePtrFromString(modelPath)
	if err != nil {
		return 0, err
	}
	backendPtr, err := utils.BytePtrFromString(backend)
	if err != nil {
		return 0, err
	}

	var visionPtr, audioPtr *byte
	if vision != nil {
		visionPtr, err = utils.BytePtrFromString(*vision)
		if err != nil {
			return 0, err
		}
	}
	if audio != nil {
		audioPtr, err = utils.BytePtrFromString(*audio)
		if err != nil {
			return 0, err
		}
	}

	return callForHandle[EngineSettings](engineSettingsCreateFunc, "engine_settings_create",
		unsafe.Pointer(&pathPtr),
		unsafe.Pointer(&backendPtr),
		unsafe.Pointer(&visionPtr),
		unsafe.Pointer(&audioPtr),
	)
}

// Delete releases an EngineSettings handle.
func (s EngineSettings) Delete() {
	if s == 0 {
		return
	}
	engineSettingsDeleteFunc.Call(nil, unsafe.Pointer(&s))
}

// SetMaxNumTokens caps the total token budget (prompt + output) for the engine.
func (s EngineSettings) SetMaxNumTokens(n int) {
	if s == 0 {
		return
	}
	engineSettingsSetMaxNumTokensFunc.Call(nil, unsafe.Pointer(&s), unsafe.Pointer(new(int32(n))))
}

// SetParallelFileSectionLoading toggles parallel loading of litertlm file
// sections (defaults to true on the C side).
func (s EngineSettings) SetParallelFileSectionLoading(on bool) {
	if s == 0 {
		return
	}
	var v uint8
	if on {
		v = 1
	}
	engineSettingsSetParallelFileSectionLoadingFunc.Call(nil, unsafe.Pointer(&s), unsafe.Pointer(&v))
}

// SetCacheDir points the engine at a directory it can use for artefact
// caching. Strings containing a NUL byte are silently ignored — they are
// invalid as C strings and could not be passed through the FFI layer.
func (s EngineSettings) SetCacheDir(dir string) {
	if s == 0 {
		return
	}
	dirPtr, err := utils.BytePtrFromString(dir)
	if err != nil {
		return
	}
	engineSettingsSetCacheDirFunc.Call(nil, unsafe.Pointer(&s), unsafe.Pointer(&dirPtr))
}

// SetActivationDataType selects the activation precision. Accepted values
// (per c/engine.h): 0=F32, 1=F16, 2=I16, 3=I8.
func (s EngineSettings) SetActivationDataType(t int) {
	if s == 0 {
		return
	}
	engineSettingsSetActivationDataTypeFunc.Call(nil, unsafe.Pointer(&s), unsafe.Pointer(new(int32(t))))
}

// SetPrefillChunkSize sets the CPU-backend prefill chunk size for dynamic models.
func (s EngineSettings) SetPrefillChunkSize(n int) {
	if s == 0 {
		return
	}
	engineSettingsSetPrefillChunkSizeFunc.Call(nil, unsafe.Pointer(&s), unsafe.Pointer(new(int32(n))))
}

// EnableBenchmark turns on benchmark collection. BenchmarkInfo is then
// retrievable via Session.BenchmarkInfo() / Conversation.BenchmarkInfo().
func (s EngineSettings) EnableBenchmark() {
	if s == 0 {
		return
	}
	engineSettingsEnableBenchmarkFunc.Call(nil, unsafe.Pointer(&s))
}

// SetNumPrefillTokens sets the number of tokens to synthesise for benchmarking prefill.
func (s EngineSettings) SetNumPrefillTokens(n int) {
	if s == 0 {
		return
	}
	engineSettingsSetNumPrefillTokensFunc.Call(nil, unsafe.Pointer(&s), unsafe.Pointer(new(int32(n))))
}

// SetNumDecodeTokens sets the number of decode steps for benchmarking.
func (s EngineSettings) SetNumDecodeTokens(n int) {
	if s == 0 {
		return
	}
	engineSettingsSetNumDecodeTokensFunc.Call(nil, unsafe.Pointer(&s), unsafe.Pointer(new(int32(n))))
}

// SetEnableSpeculativeDecoding toggles speculative decoding on the engine.
func (s EngineSettings) SetEnableSpeculativeDecoding(on bool) {
	if s == 0 {
		return
	}
	var v uint8
	if on {
		v = 1
	}
	engineSettingsSetEnableSpeculativeDecodingFunc.Call(nil, unsafe.Pointer(&s), unsafe.Pointer(&v))
}

// SetMaxNumImages caps the maximum number of image inputs the engine
// reserves capacity for. Honored only by the legacy engine
// implementation per the upstream C header.
func (s EngineSettings) SetMaxNumImages(n int) {
	if s == 0 {
		return
	}
	engineSettingsSetMaxNumImagesFunc.Call(nil, unsafe.Pointer(&s), unsafe.Pointer(new(int32(n))))
}

// SetLitertDispatchLibDir sets the LiteRT dispatch library directory
// used by the NPU backend. Ignored by CPU / GPU backends.
func (s EngineSettings) SetLitertDispatchLibDir(dir string) {
	if s == 0 {
		return
	}
	dirPtr, err := utils.BytePtrFromString(dir)
	if err != nil {
		return
	}
	engineSettingsSetLitertDispatchLibDirFunc.Call(nil, unsafe.Pointer(&s), unsafe.Pointer(&dirPtr))
}

// NewEngine loads the model described by settings. The caller still owns
// the settings handle and may Delete() it once the engine is created.
func NewEngine(settings EngineSettings) (Engine, error) {
	if settings == 0 {
		return 0, fmt.Errorf("litertlm: engine_create: settings is nil")
	}
	return callForHandle[Engine](engineCreateFunc, "engine_create", unsafe.Pointer(&settings))
}

// Delete releases an Engine handle and frees the underlying model weights.
func (e Engine) Delete() {
	if e == 0 {
		return
	}
	engineDeleteFunc.Call(nil, unsafe.Pointer(&e))
}

// NewSession opens a session on the engine. Pass 0 for cfg to accept defaults.
func (e Engine) NewSession(cfg SessionConfig) (Session, error) {
	if e == 0 {
		return 0, fmt.Errorf("litertlm: engine_create_session: invalid engine")
	}
	return callForHandle[Session](engineCreateSessionFunc, "engine_create_session",
		unsafe.Pointer(&e),
		unsafe.Pointer(&cfg),
	)
}

// NewConversation creates a conversation rooted in the engine.
func (e Engine) NewConversation(cfg ConversationConfig) (Conversation, error) {
	if e == 0 {
		return 0, fmt.Errorf("litertlm: conversation_create: invalid engine")
	}
	return callForHandle[Conversation](conversationCreateFunc, "conversation_create",
		unsafe.Pointer(&e),
		unsafe.Pointer(&cfg),
	)
}
