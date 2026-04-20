package litertlm

import (
	"errors"
	"runtime"
	"unsafe"
)

// NewSessionConfig creates a fresh SessionConfig with C-side defaults. The
// caller owns the handle and must invoke Delete() when done (after the
// Session that consumed it has also been created — the C API copies the
// relevant fields).
func NewSessionConfig() (SessionConfig, error) {
	var c SessionConfig
	sessionConfigCreateFunc.Call(unsafe.Pointer(&c))
	if c == 0 {
		return 0, errors.New("litertlm: session_config_create failed")
	}
	return c, nil
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
	v := int32(n)
	sessionConfigSetMaxOutputTokensFunc.Call(nil, unsafe.Pointer(&c), unsafe.Pointer(&v))
}

// SetSamplerParams attaches sampler parameters to the session config. The
// parameters are read by C during this call, so the Go value does not need
// to outlive the call.
func (c SessionConfig) SetSamplerParams(p SamplerParams) {
	if c == 0 {
		return
	}
	pPtr := unsafe.Pointer(&p)
	sessionConfigSetSamplerParamsFunc.Call(nil, unsafe.Pointer(&c), unsafe.Pointer(&pPtr))
	runtime.KeepAlive(p)
}

// Delete releases a Session handle.
func (s Session) Delete() {
	if s == 0 {
		return
	}
	sessionDeleteFunc.Call(nil, unsafe.Pointer(&s))
}

// GenerateContent runs synchronous inference for the given multimodal inputs
// and returns a Responses handle. The caller must Delete() the returned
// Responses when finished with it.
func (s Session) GenerateContent(inputs []InputData) (Responses, error) {
	if s == 0 {
		return 0, errors.New("litertlm: generate_content: invalid session")
	}
	if len(inputs) == 0 {
		return 0, errors.New("litertlm: generate_content: no inputs")
	}

	inputsPtr := unsafe.Pointer(&inputs[0])
	n := uint64(len(inputs))

	var r Responses
	sessionGenerateContentFunc.Call(
		unsafe.Pointer(&r),
		unsafe.Pointer(&s),
		unsafe.Pointer(&inputsPtr),
		unsafe.Pointer(&n),
	)
	// Keep the inputs slice (and any byte buffers it references) alive until
	// C has finished reading them. This is a synchronous call, so the slice
	// only needs to survive the Call above, but KeepAlive makes the guarantee
	// explicit for readers auditing GC safety.
	runtime.KeepAlive(inputs)

	if r == 0 {
		return 0, errors.New("litertlm: generate_content failed")
	}
	return r, nil
}

// BenchmarkInfo retrieves benchmark data collected for this session's
// generations. Requires EngineSettings.EnableBenchmark() to have been set
// on the engine at construction time. The returned handle must be deleted.
func (s Session) BenchmarkInfo() (BenchmarkInfo, error) {
	if s == 0 {
		return 0, errors.New("litertlm: benchmark_info: invalid session")
	}
	var b BenchmarkInfo
	sessionGetBenchmarkInfoFunc.Call(unsafe.Pointer(&b), unsafe.Pointer(&s))
	if b == 0 {
		return 0, errors.New("litertlm: session_get_benchmark_info failed")
	}
	return b, nil
}
