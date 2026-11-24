package litertlm

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/vladimirvivien/litertlm-go/internal/cstr"
)

// NewSessionConfig creates a fresh SessionConfig with C-side defaults.
// Caller must Delete() once the consuming Session has been created (the
// C API copies the relevant fields out).
func NewSessionConfig() (SessionConfig, error) {
	return callStatusHandle[SessionConfig](sessionConfigCreateFunc, "session_config_create")
}

// Delete releases a SessionConfig handle.
func (c SessionConfig) Delete() { deleteHandle(c, sessionConfigDeleteFunc) }

// SetMaxOutputTokens caps the output tokens produced per decode step.
func (c SessionConfig) SetMaxOutputTokens(n int) error {
	return setHandleScalar(c, "session config", sessionConfigSetMaxOutputTokensFunc, "session_config_set_max_output_tokens", int32(n))
}

// SetApplyPromptTemplate toggles whether session-level inputs run through
// the model's prompt template before reaching the engine.
func (c SessionConfig) SetApplyPromptTemplate(on bool) error {
	return setHandleBool(c, "session config", sessionConfigSetApplyPromptTemplateFunc, "session_config_set_apply_prompt_template", on)
}

// SetSamplerParams attaches sampler parameters to the session config.
func (c SessionConfig) SetSamplerParams(p SamplerParams) error {
	if c == 0 {
		return fmt.Errorf("litertlm: set_sampler_params: invalid session config")
	}
	return withTempHandle(
		func() (samplerParamsHandle, error) { return newSamplerParamsHandle(p) },
		func(h samplerParamsHandle) { h.delete() },
		func(h samplerParamsHandle) error {
			return setHandleConfig(c, "session config", sessionConfigSetSamplerParamsFunc, "session_config_set_sampler_params", h)
		},
	)
}

// SetSpeculativeDecodingConfig overrides the engine's speculative decoding
// configuration for sessions constructed with this config.
func (c SessionConfig) SetSpeculativeDecodingConfig(cfg SpeculativeDecodingConfig) error {
	return setSpeculativeDecodingConfig(c, "session config", sessionConfigSetSpeculativeDecodingConfigFunc,
		"session_config_set_speculative_decoding_config", cfg)
}

// SetLoRAPath sets the path to the text LoRA weights file for this session.
func (c SessionConfig) SetLoRAPath(path string) error {
	return setHandleString(c, "session config", sessionConfigSetLoraPathFunc, "session_config_set_lora_path", path)
}

// SetAudioLoRAPath sets the path to the audio LoRA weights file for this session.
func (c SessionConfig) SetAudioLoRAPath(path string) error {
	return setHandleString(c, "session config", sessionConfigSetAudioLoraPathFunc, "session_config_set_audio_lora_path", path)
}

// Delete releases a Session handle.
func (s Session) Delete() { deleteHandle(s, sessionDeleteFunc) }

// Clone returns an independent copy of s carrying its current prefill and KV state.
// Caller must Delete() the cloned Session.
func (s Session) Clone() (Session, error) {
	if s == 0 {
		return 0, fmt.Errorf("litertlm: session_clone: invalid session")
	}
	return callStatusHandle[Session](sessionCloneFunc, "session_clone", unsafe.Pointer(&s))
}

// SaveCheckpoint records the session's current state under label for later rewind.
func (s Session) SaveCheckpoint(label string) error {
	return setHandleString(s, "session", sessionSaveCheckpointFunc, "session_save_checkpoint", label)
}

// RewindToCheckpoint restores the session state saved under label.
func (s Session) RewindToCheckpoint(label string) error {
	return setHandleString(s, "session", sessionRewindToCheckpointFunc, "session_rewind_to_checkpoint", label)
}

// RewindToStep restores the session state to the given token step number.
func (s Session) RewindToStep(step int) error {
	return setHandleScalar(s, "session", sessionRewindToStepFunc, "session_rewind_to_step", int32(step))
}

// GenerateContent runs synchronous inference for the given multimodal
// inputs. Caller must Delete() the returned Responses.
func (s Session) GenerateContent(inputs []InputData) (Responses, error) {
	if s == 0 {
		return 0, fmt.Errorf("litertlm: generate_content: invalid session")
	}
	if len(inputs) == 0 {
		return 0, fmt.Errorf("litertlm: generate_content: no inputs")
	}

	ptrs := handlePtrs(inputs)
	inputsPtr := unsafe.Pointer(&ptrs[0])
	n := uint64(len(inputs))

	r, err := callStatusHandle[Responses](sessionGenerateContentFunc, "session_generate_content",
		unsafe.Pointer(&s),
		unsafe.Pointer(&inputsPtr),
		unsafe.Pointer(&n),
	)
	runtime.KeepAlive(ptrs)
	return r, err
}

// BenchmarkInfo retrieves benchmark data collected for this session's
// generations. Requires EngineSettings.EnableBenchmark() at engine
// construction time. Caller must Delete().
func (s Session) BenchmarkInfo() (BenchmarkInfo, error) {
	if s == 0 {
		return 0, fmt.Errorf("litertlm: benchmark_info: invalid session")
	}
	return callStatusHandle[BenchmarkInfo](sessionGetBenchmarkInfoFunc,
		"session_get_benchmark_info", unsafe.Pointer(&s))
}

// Cancel requests cancellation of an in-flight process on this session.
// It is safe to call when no process is running.
func (s Session) Cancel() {
	if s == 0 {
		return
	}
	var status int32
	_ = sessionCancelProcessFunc.callErr(unsafe.Pointer(&status), unsafe.Pointer(&s))
}

// RunPrefill adds inputs to the session's context as the prompt phase.
// Pair with RunDecode to drive the explicit prefill→decode flow rather
// than the all-in-one GenerateContent.
func (s Session) RunPrefill(inputs []InputData) error {
	if s == 0 {
		return fmt.Errorf("litertlm: session_run_prefill: invalid session")
	}
	if len(inputs) == 0 {
		return fmt.Errorf("litertlm: session_run_prefill: no inputs")
	}

	ptrs := handlePtrs(inputs)
	inputsPtr := unsafe.Pointer(&ptrs[0])
	n := uint64(len(inputs))

	err := callStatus(sessionRunPrefillFunc, "session_run_prefill",
		unsafe.Pointer(&s),
		unsafe.Pointer(&inputsPtr),
		unsafe.Pointer(&n),
	)
	runtime.KeepAlive(ptrs)
	return err
}

// RunDecode performs the decode phase against context that's been
// populated via RunPrefill. Caller must Delete() the returned Responses.
func (s Session) RunDecode() (Responses, error) {
	if s == 0 {
		return 0, fmt.Errorf("litertlm: session_run_decode: invalid session")
	}
	return callStatusHandle[Responses](sessionRunDecodeFunc, "session_run_decode",
		unsafe.Pointer(&s))
}

// ScoreTexts scores candidate strings against the prefilled prompt. Pass one
// target, or multiple single-token targets (C API 1.1.0+) scored from one
// decode step. storeTokenLengths attaches tokenized target lengths (read via
// Responses.TokenLength). Because scoring advances model state, rewind with
// SaveCheckpoint/RewindToCheckpoint before further prefill or decode. Caller
// must Delete() the returned Responses.
func (s Session) ScoreTexts(targets []string, storeTokenLengths bool) (Responses, error) {
	if s == 0 {
		return 0, fmt.Errorf("litertlm: session_run_text_scoring: invalid session")
	}
	if len(targets) == 0 {
		return 0, fmt.Errorf("litertlm: session_run_text_scoring: no targets")
	}

	ptrs := make([]*byte, len(targets))
	for i, t := range targets {
		p, err := cstr.BytePtrFromString(t)
		if err != nil {
			return 0, err
		}
		ptrs[i] = p
	}
	arrPtr := unsafe.Pointer(&ptrs[0])
	num := uint64(len(targets))
	var enable uint8
	if storeTokenLengths {
		enable = 1
	}

	r, err := callStatusHandle[Responses](sessionRunTextScoringFunc, "session_run_text_scoring",
		unsafe.Pointer(&s),
		unsafe.Pointer(&arrPtr),
		unsafe.Pointer(&num),
		unsafe.Pointer(&enable),
	)
	runtime.KeepAlive(ptrs)
	return r, err
}
