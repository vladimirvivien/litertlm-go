package litertlm

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/vladimirvivien/litertlm-go/pkg/utils"
)

// NewSessionConfig creates a fresh SessionConfig with C-side defaults.
// Caller must Delete() once the consuming Session has been created (the
// C API copies the relevant fields out).
func NewSessionConfig() (SessionConfig, error) {
	return callForHandle[SessionConfig](sessionConfigCreateFunc, "session_config_create")
}

// Delete releases a SessionConfig handle.
func (c SessionConfig) Delete() {
	if c == 0 {
		return
	}
	sessionConfigDeleteFunc.Call(nil, unsafe.Pointer(&c))
}

// SetMaxOutputTokens caps the output tokens produced per decode step.
func (c SessionConfig) SetMaxOutputTokens(n int) {
	if c == 0 {
		return
	}
	sessionConfigSetMaxOutputTokensFunc.Call(nil, unsafe.Pointer(&c), unsafe.Pointer(new(int32(n))))
}

// SetApplyPromptTemplate toggles whether session-level inputs run through
// the model's prompt template before reaching the engine.
func (c SessionConfig) SetApplyPromptTemplate(on bool) {
	if c == 0 {
		return
	}
	var v uint8
	if on {
		v = 1
	}
	sessionConfigSetApplyPromptTemplateFunc.Call(nil, unsafe.Pointer(&c), unsafe.Pointer(&v))
}

// SetSamplerParams attaches sampler parameters to the session config.
func (c SessionConfig) SetSamplerParams(p SamplerParams) {
	if c == 0 {
		return
	}
	pPtr := unsafe.Pointer(&p)
	sessionConfigSetSamplerParamsFunc.Call(nil, unsafe.Pointer(&c), unsafe.Pointer(&pPtr))
	// KeepAlive after Call ensures the compiler doesn't drop p before the
	// C side has finished reading through pPtr.
	runtime.KeepAlive(p)
}

// Delete releases a Session handle.
func (s Session) Delete() {
	if s == 0 {
		return
	}
	sessionDeleteFunc.Call(nil, unsafe.Pointer(&s))
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

	inputsPtr := unsafe.Pointer(&inputs[0])
	n := uint64(len(inputs))

	r, err := callForHandle[Responses](sessionGenerateContentFunc, "generate_content",
		unsafe.Pointer(&s),
		unsafe.Pointer(&inputsPtr),
		unsafe.Pointer(&n),
	)
	// KeepAlive after the Call ensures inputs (and the byte buffers it
	// references via &inputs[0]) survive until C is done reading them.
	runtime.KeepAlive(inputs)
	return r, err
}

// BenchmarkInfo retrieves benchmark data collected for this session's
// generations. Requires EngineSettings.EnableBenchmark() at engine
// construction time. Caller must Delete().
func (s Session) BenchmarkInfo() (BenchmarkInfo, error) {
	if s == 0 {
		return 0, fmt.Errorf("litertlm: benchmark_info: invalid session")
	}
	return callForHandle[BenchmarkInfo](sessionGetBenchmarkInfoFunc,
		"session_get_benchmark_info", unsafe.Pointer(&s))
}

// Cancel requests cancellation of an in-flight process on this session.
// It is safe to call when no process is running.
func (s Session) Cancel() {
	if s == 0 {
		return
	}
	sessionCancelProcessFunc.Call(nil, unsafe.Pointer(&s))
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

	inputsPtr := unsafe.Pointer(&inputs[0])
	n := uint64(len(inputs))

	var status int32
	sessionRunPrefillFunc.Call(
		unsafe.Pointer(&status),
		unsafe.Pointer(&s),
		unsafe.Pointer(&inputsPtr),
		unsafe.Pointer(&n),
	)
	runtime.KeepAlive(inputs)
	if status != 0 {
		return fmt.Errorf("litertlm: session_run_prefill failed (code=%d)", status)
	}
	return nil
}

// RunDecode performs the decode phase against context that's been
// populated via RunPrefill. Caller must Delete() the returned Responses.
func (s Session) RunDecode() (Responses, error) {
	if s == 0 {
		return 0, fmt.Errorf("litertlm: session_run_decode: invalid session")
	}
	return callForHandle[Responses](sessionRunDecodeFunc, "session_run_decode",
		unsafe.Pointer(&s))
}

// ScoreTexts scores each candidate string against the prefilled prompt.
// storeTokenLengths attaches the per-target tokenized length to the
// returned Responses (retrievable via Responses.TokenLength once that
// accessor lands in 4d). Caller must Delete() the result.
func (s Session) ScoreTexts(targets []string, storeTokenLengths bool) (Responses, error) {
	if s == 0 {
		return 0, fmt.Errorf("litertlm: session_run_text_scoring: invalid session")
	}
	if len(targets) == 0 {
		return 0, fmt.Errorf("litertlm: session_run_text_scoring: no targets")
	}

	// Build a contiguous []*byte so &ptrs[0] gives C its const char**.
	ptrs := make([]*byte, len(targets))
	for i, t := range targets {
		p, err := utils.BytePtrFromString(t)
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

	r, err := callForHandle[Responses](sessionRunTextScoringFunc, "session_run_text_scoring",
		unsafe.Pointer(&s),
		unsafe.Pointer(&arrPtr),
		unsafe.Pointer(&num),
		unsafe.Pointer(&enable),
	)
	// Hold the per-target byte buffers alive across the FFI call.
	runtime.KeepAlive(ptrs)
	return r, err
}
