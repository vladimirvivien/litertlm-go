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
	h, err := newSamplerParamsHandle(p)
	if err != nil {
		return err
	}
	defer h.delete()

	err = callStatus(sessionConfigSetSamplerParamsFunc, "session_config_set_sampler_params",
		unsafe.Pointer(&c), unsafe.Pointer(&h))
	runtime.KeepAlive(h)
	return err
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

// ScoreTexts scores each candidate string against the prefilled
// prompt. storeTokenLengths attaches the per-target tokenized length
// to the returned Responses (read via Responses.TokenLength). Caller
// must Delete() the result.
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
