package litertlm

import (
	"fmt"
	"runtime"
	"unsafe"
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
