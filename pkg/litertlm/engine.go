package litertlm

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/vladimirvivien/litertlm-go/internal/cstr"
)

// NewEngineSettings constructs an EngineSettings handle. vision and audio
// may be nil to leave those backends unset. Caller must Delete().
func NewEngineSettings(modelPath, backend string, vision, audio *string) (EngineSettings, error) {
	pathPtr, err := cstr.BytePtrFromString(modelPath)
	if err != nil {
		return 0, err
	}
	backendPtr, visionPtr, audioPtr, err := marshalBackendPtrs(backend, vision, audio)
	if err != nil {
		return 0, err
	}

	s, err := callStatusHandle[EngineSettings](engineSettingsCreateFunc, "engine_settings_create",
		unsafe.Pointer(&pathPtr),
		unsafe.Pointer(&backendPtr),
		unsafe.Pointer(&visionPtr),
		unsafe.Pointer(&audioPtr),
	)
	runtime.KeepAlive(pathPtr)
	runtime.KeepAlive(backendPtr)
	runtime.KeepAlive(visionPtr)
	runtime.KeepAlive(audioPtr)
	return s, err
}

// NewEngineSettingsFromFD constructs an EngineSettings handle from a raw file descriptor.
// The engine takes ownership of the file descriptor and will close it when done.
// vision and audio may be nil to leave those backends unset. Caller must Delete().
func NewEngineSettingsFromFD(fd int, backend string, vision, audio *string) (EngineSettings, error) {
	backendPtr, visionPtr, audioPtr, err := marshalBackendPtrs(backend, vision, audio)
	if err != nil {
		return 0, err
	}

	fdVal := int32(fd)
	s, err := callStatusHandle[EngineSettings](engineSettingsCreateFromRawFileDescriptorFunc, "engine_settings_create_from_raw_file_descriptor",
		unsafe.Pointer(&fdVal),
		unsafe.Pointer(&backendPtr),
		unsafe.Pointer(&visionPtr),
		unsafe.Pointer(&audioPtr),
	)
	runtime.KeepAlive(backendPtr)
	runtime.KeepAlive(visionPtr)
	runtime.KeepAlive(audioPtr)
	return s, err
}

// Delete releases an EngineSettings handle.
func (s EngineSettings) Delete() { deleteHandle(s, engineSettingsDeleteFunc) }

// SetMaxNumTokens caps the total token budget (prompt + output) for the engine.
func (s EngineSettings) SetMaxNumTokens(n int) error {
	return setHandleScalar(s, "settings", engineSettingsSetMaxNumTokensFunc, "engine_settings_set_max_num_tokens", int32(n))
}

// SetParallelFileSectionLoading toggles parallel loading of litertlm file
// sections (defaults to true on the C side).
func (s EngineSettings) SetParallelFileSectionLoading(on bool) error {
	return setHandleBool(s, "settings", engineSettingsSetParallelFileSectionLoadingFunc, "engine_settings_set_parallel_file_section_loading", on)
}

// SetCacheDir points the engine at a directory it can use for artefact caching.
func (s EngineSettings) SetCacheDir(dir string) error {
	return setHandleString(s, "settings", engineSettingsSetCacheDirFunc, "engine_settings_set_cache_dir", dir)
}

// SetActivationDataType selects the activation precision. Accepted values
// (per c/engine.h): 0=F32, 1=F16, 2=I16, 3=I8.
func (s EngineSettings) SetActivationDataType(t int) error {
	return setHandleScalar(s, "settings", engineSettingsSetActivationDataTypeFunc, "engine_settings_set_activation_data_type", int32(t))
}

// SetPrefillChunkSize sets the CPU-backend prefill chunk size for dynamic models.
func (s EngineSettings) SetPrefillChunkSize(n int) error {
	return setHandleScalar(s, "settings", engineSettingsSetPrefillChunkSizeFunc, "engine_settings_set_prefill_chunk_size", int32(n))
}

// EnableBenchmark turns on benchmark collection. BenchmarkInfo is then
// retrievable via Session.BenchmarkInfo() / Conversation.BenchmarkInfo().
func (s EngineSettings) EnableBenchmark() error {
	if s == 0 {
		return fmt.Errorf("litertlm: engine_settings_enable_benchmark: invalid settings")
	}
	return callStatus(engineSettingsEnableBenchmarkFunc, "engine_settings_enable_benchmark", unsafe.Pointer(&s))
}

// SetNumPrefillTokens sets the number of tokens to synthesise for benchmarking prefill.
func (s EngineSettings) SetNumPrefillTokens(n int) error {
	return setHandleScalar(s, "settings", engineSettingsSetNumPrefillTokensFunc, "engine_settings_set_num_prefill_tokens", int32(n))
}

// SetNumDecodeTokens sets the number of decode steps for benchmarking.
func (s EngineSettings) SetNumDecodeTokens(n int) error {
	return setHandleScalar(s, "settings", engineSettingsSetNumDecodeTokensFunc, "engine_settings_set_num_decode_tokens", int32(n))
}

// SetEnableSpeculativeDecoding toggles speculative decoding on the engine.
func (s EngineSettings) SetEnableSpeculativeDecoding(on bool) error {
	return setHandleBool(s, "settings", engineSettingsSetEnableSpeculativeDecodingFunc, "engine_settings_set_enable_speculative_decoding", on)
}

// SetMaxNumImages caps the maximum number of image inputs the engine
// reserves capacity for. Honored only by the legacy engine
// implementation per the upstream C header.
func (s EngineSettings) SetMaxNumImages(n int) error {
	return setHandleScalar(s, "settings", engineSettingsSetMaxNumImagesFunc, "engine_settings_set_max_num_images", int32(n))
}

// SetLitertDispatchLibDir sets the LiteRT dispatch library directory
// used by the NPU backend. Ignored by CPU / GPU backends.
func (s EngineSettings) SetLitertDispatchLibDir(dir string) error {
	return setHandleString(s, "settings", engineSettingsSetLitertDispatchLibDirFunc, "engine_settings_set_litert_dispatch_lib_dir", dir)
}

// SetNumThreads sets the number of threads for the CPU backend.
func (s EngineSettings) SetNumThreads(n int) error {
	return setHandleScalar(s, "settings", engineSettingsSetNumThreadsFunc, "engine_settings_set_num_threads", int32(n))
}

// SetAudioNumThreads sets the number of threads for the audio CPU backend.
func (s EngineSettings) SetAudioNumThreads(n int) error {
	return setHandleScalar(s, "settings", engineSettingsSetAudioNumThreadsFunc, "engine_settings_set_audio_num_threads", int32(n))
}

// SetLoRARank sets the LoRA rank for the engine.
func (s EngineSettings) SetLoRARank(rank int) error {
	return setHandleScalar(s, "settings", engineSettingsSetLoraRankFunc, "engine_settings_set_lora_rank", int32(rank))
}

// SetSupportedLoRARanks sets the supported LoRA ranks for the engine.
func (s EngineSettings) SetSupportedLoRARanks(ranks []int) error {
	return setHandleInt32Slice(s, "settings", engineSettingsSetSupportedLoraRanksFunc, "engine_settings_set_supported_lora_ranks", ranks)
}

// SetAudioLoRARank sets the audio LoRA rank for the engine.
func (s EngineSettings) SetAudioLoRARank(rank int) error {
	return setHandleScalar(s, "settings", engineSettingsSetAudioLoraRankFunc, "engine_settings_set_audio_lora_rank", int32(rank))
}

// SetSupportedAudioLoRARanks sets the supported audio LoRA ranks for the engine.
func (s EngineSettings) SetSupportedAudioLoRARanks(ranks []int) error {
	return setHandleInt32Slice(s, "settings", engineSettingsSetSupportedAudioLoraRanksFunc, "engine_settings_set_supported_audio_lora_ranks", ranks)
}

// SetSpeculativeDecodingConfig configures default speculative decoding mode and
// draft lenience for all sessions created by this engine.
func (s EngineSettings) SetSpeculativeDecodingConfig(cfg SpeculativeDecodingConfig) error {
	return setSpeculativeDecodingConfig(s, "settings", engineSettingsSetSpeculativeDecodingConfigFunc,
		"engine_settings_set_speculative_decoding_config", cfg)
}

// SetEnableYNNPACK enables YNNPACK CPU delegation before XNNPACK.
func (s EngineSettings) SetEnableYNNPACK(on bool) error {
	return setHandleBool(s, "settings", engineSettingsSetEnableYnnpackFunc, "engine_settings_set_enable_ynnpack", on)
}

// SetMaxVisionTokensPerImage caps vision tokens generated per image.
func (s EngineSettings) SetMaxVisionTokensPerImage(n int) error {
	return setHandleScalar(s, "settings", engineSettingsSetMaxVisionTokensPerImageFunc, "engine_settings_set_max_vision_tokens_per_image", int32(n))
}

// SetGPUEnableMetalResidencySet enables Metal residency sets on Apple GPU backends.
func (s EngineSettings) SetGPUEnableMetalResidencySet(on bool) error {
	return setHandleBool(s, "settings", engineSettingsSetGpuEnableMetalResidencySetFunc, "engine_settings_set_gpu_enable_metal_residency_set", on)
}

// SetGPUDecodeStepsPerSync configures decode steps between GPU syncs.
func (s EngineSettings) SetGPUDecodeStepsPerSync(n int) error {
	return setHandleScalar(s, "settings", engineSettingsSetGpuDecodeStepsPerSyncFunc, "engine_settings_set_gpu_decode_steps_per_sync", int32(n))
}

// SetGPUWaitForWeightUploads configures whether GPU initialization waits for weight uploads.
func (s EngineSettings) SetGPUWaitForWeightUploads(on bool) error {
	return setHandleBool(s, "settings", engineSettingsSetGpuWaitForWeightUploadsFunc, "engine_settings_set_gpu_wait_for_weight_uploads", on)
}

// SetUseRingbuffersLocalAttention toggles ringbuffer KV storage for local attention layers.
func (s EngineSettings) SetUseRingbuffersLocalAttention(on bool) error {
	return setHandleBool(s, "settings", engineSettingsSetUseRingbuffersLocalAttentionFunc, "engine_settings_set_use_ringbuffers_local_attention", on)
}

// SetSingleThreadedExecution runs session execution on caller-driven threads rather than background workers.
func (s EngineSettings) SetSingleThreadedExecution(on bool) error {
	return setHandleBool(s, "settings", engineSettingsSetSingleThreadedExecutionFunc, "engine_settings_set_single_threaded_execution", on)
}

// SetSelectedSignatures restricts the main executor to the listed model signatures
// (requires LiteRT-LM C API 1.1.0+).
func (s EngineSettings) SetSelectedSignatures(signatures []string) error {
	if err := requireAPIVersion(1, 1); err != nil {
		return err
	}
	return setHandleStringSlice(s, "settings", engineSettingsSetSelectedSignaturesFunc, "engine_settings_set_selected_signatures", signatures)
}

// NewEngine loads the model described by settings. The caller still owns
// the settings handle and may Delete() it once the engine is created.
func NewEngine(settings EngineSettings) (Engine, error) {
	if settings == 0 {
		return 0, fmt.Errorf("litertlm: engine_create: invalid settings")
	}
	return callStatusHandle[Engine](engineCreateFunc, "engine_create", unsafe.Pointer(&settings))
}

// Delete releases an Engine handle and frees the underlying model weights.
func (e Engine) Delete() { deleteHandle(e, engineDeleteFunc) }

// NewSession opens a session on the engine. Pass 0 for cfg to accept defaults.
func (e Engine) NewSession(cfg SessionConfig) (Session, error) {
	if e == 0 {
		return 0, fmt.Errorf("litertlm: engine_create_session: invalid engine")
	}
	return callStatusHandle[Session](engineCreateSessionFunc, "engine_create_session",
		unsafe.Pointer(&e),
		unsafe.Pointer(&cfg),
	)
}

// NewConversation creates a conversation rooted in the engine.
func (e Engine) NewConversation(cfg ConversationConfig) (Conversation, error) {
	if e == 0 {
		return 0, fmt.Errorf("litertlm: conversation_create: invalid engine")
	}
	return callStatusHandle[Conversation](conversationCreateFunc, "conversation_create",
		unsafe.Pointer(&e),
		unsafe.Pointer(&cfg),
	)
}
