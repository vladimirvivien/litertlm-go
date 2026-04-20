package litertlm

import (
	"errors"
	"unsafe"

	"github.com/vladimirvivien/litertlm-go/pkg/utils"
)

// NewEngineSettings constructs an EngineSettings handle. modelPath is the
// path to the .litertlm model file. backend is typically "cpu" or "gpu".
// vision and audio are optional extra backends; pass nil to leave them unset.
//
// The returned handle must be released with Delete() once an Engine has been
// created from it (or if creation failed).
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

	var s EngineSettings
	engineSettingsCreateFunc.Call(
		unsafe.Pointer(&s),
		unsafe.Pointer(&pathPtr),
		unsafe.Pointer(&backendPtr),
		unsafe.Pointer(&visionPtr),
		unsafe.Pointer(&audioPtr),
	)
	if s == 0 {
		return 0, errors.New("litertlm: engine_settings_create failed")
	}
	return s, nil
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
	v := int32(n)
	engineSettingsSetMaxNumTokensFunc.Call(nil, unsafe.Pointer(&s), unsafe.Pointer(&v))
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

// SetCacheDir points the engine at a directory it can use for artefact caching.
func (s EngineSettings) SetCacheDir(dir string) error {
	if s == 0 {
		return nil
	}
	dirPtr, err := utils.BytePtrFromString(dir)
	if err != nil {
		return err
	}
	engineSettingsSetCacheDirFunc.Call(nil, unsafe.Pointer(&s), unsafe.Pointer(&dirPtr))
	return nil
}

// SetActivationDataType selects the activation precision. Accepted values
// (per c/engine.h): 0=F32, 1=F16, 2=I16, 3=I8.
func (s EngineSettings) SetActivationDataType(t int) {
	if s == 0 {
		return
	}
	v := int32(t)
	engineSettingsSetActivationDataTypeFunc.Call(nil, unsafe.Pointer(&s), unsafe.Pointer(&v))
}

// SetPrefillChunkSize sets the CPU-backend prefill chunk size for dynamic models.
func (s EngineSettings) SetPrefillChunkSize(n int) {
	if s == 0 {
		return
	}
	v := int32(n)
	engineSettingsSetPrefillChunkSizeFunc.Call(nil, unsafe.Pointer(&s), unsafe.Pointer(&v))
}

// EnableBenchmark turns on benchmark collection for the engine. BenchmarkInfo
// can then be retrieved via Session.BenchmarkInfo() / Conversation.BenchmarkInfo().
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
	v := int32(n)
	engineSettingsSetNumPrefillTokensFunc.Call(nil, unsafe.Pointer(&s), unsafe.Pointer(&v))
}

// SetNumDecodeTokens sets the number of decode steps for benchmarking.
func (s EngineSettings) SetNumDecodeTokens(n int) {
	if s == 0 {
		return
	}
	v := int32(n)
	engineSettingsSetNumDecodeTokensFunc.Call(nil, unsafe.Pointer(&s), unsafe.Pointer(&v))
}

// NewEngine loads the model described by the given settings and returns a
// live Engine. The caller retains ownership of the settings handle — it is
// safe (and recommended) to call settings.Delete() once the engine is created.
func NewEngine(settings EngineSettings) (Engine, error) {
	if settings == 0 {
		return 0, errors.New("litertlm: engine_create: settings is nil")
	}
	var e Engine
	engineCreateFunc.Call(unsafe.Pointer(&e), unsafe.Pointer(&settings))
	if e == 0 {
		return 0, errors.New("litertlm: engine_create failed")
	}
	return e, nil
}

// Delete releases an Engine handle and frees the underlying model weights.
func (e Engine) Delete() {
	if e == 0 {
		return
	}
	engineDeleteFunc.Call(nil, unsafe.Pointer(&e))
}

// NewSession opens a session on the engine. Pass a SessionConfig with tuned
// sampler / max-output-tokens, or 0 to accept defaults.
func (e Engine) NewSession(cfg SessionConfig) (Session, error) {
	if e == 0 {
		return 0, errors.New("litertlm: engine_create_session: invalid engine")
	}
	var sess Session
	engineCreateSessionFunc.Call(
		unsafe.Pointer(&sess),
		unsafe.Pointer(&e),
		unsafe.Pointer(&cfg),
	)
	if sess == 0 {
		return 0, errors.New("litertlm: engine_create_session failed")
	}
	return sess, nil
}

// NewConversation creates a conversation rooted in the engine.
func (e Engine) NewConversation(cfg ConversationConfig) (Conversation, error) {
	if e == 0 {
		return 0, errors.New("litertlm: conversation_create: invalid engine")
	}
	var c Conversation
	conversationCreateFunc.Call(
		unsafe.Pointer(&c),
		unsafe.Pointer(&e),
		unsafe.Pointer(&cfg),
	)
	if c == 0 {
		return 0, errors.New("litertlm: conversation_create failed")
	}
	return c, nil
}
