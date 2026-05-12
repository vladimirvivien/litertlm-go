package litertlm

// Option configures a Client at New time.
type Option func(*clientConfig)

// GenOption tunes a single Generate / GenerateStream call without
// rebuilding the Client.
type GenOption func(*genConfig)

// clientConfig is the resolved configuration produced by composing
// defaults, environment fallbacks, and user-supplied Options.
type clientConfig struct {
	libPath       string
	libName       string
	modelPath     string
	backend       string
	visionBackend *string
	audioBackend  *string

	maxTokens          int
	cacheDir           string
	activationDataType *int
	prefillChunkSize   *int

	// *bool fields stay nil until the user opts in or out, preserving
	// the C-side default when no Option sets them.
	speculativeDecodingEnabled *bool
	benchmarkEnabled           *bool
	parallelFileLoading        *bool

	maxNumImages   *int
	dispatchLibDir string

	logLevel    int
	logLevelSet bool

	defaultSampler *SamplerParams
}

// genConfig is the resolved per-call configuration.
type genConfig struct {
	maxOutputTokens int
	sampler         *SamplerParams

	// GenerateData-specific. Generate ignores these fields.
	retries           int
	schemaInstruction string
}

// ---- constructor options ----

// WithLib sets the directory holding the LiteRT-LM shared libraries.
// Empty string falls back to the LITERTLM_LIB environment variable,
// then the platform default paths (see loader.DefaultPaths).
func WithLib(dir string) Option { return func(c *clientConfig) { c.libPath = dir } }

// WithLibName overrides the main C-API library short-name. By default
// the name is selected by backend ("litertlm_c_cpu" for cpu,
// "litertlm_c" for gpu). Set this when your build produces a library
// with a non-standard name (e.g. "litertlm_c_arm64"). Empty string
// falls back to the LITERTLM_LIB_NAME environment variable, then the
// backend default.
func WithLibName(name string) Option { return func(c *clientConfig) { c.libName = name } }

// WithModel sets the path to the .litertlm model file. Empty string
// falls back to the LITERTLM_MODEL environment variable.
func WithModel(path string) Option { return func(c *clientConfig) { c.modelPath = path } }

// WithBackend selects the inference backend ("cpu" or "gpu"). Default
// is "cpu" when unset.
func WithBackend(b string) Option { return func(c *clientConfig) { c.backend = b } }

// WithVisionBackend selects an optional vision backend. Pass an empty
// string to leave unset.
func WithVisionBackend(b string) Option {
	return func(c *clientConfig) {
		if b == "" {
			c.visionBackend = nil
			return
		}
		bb := b
		c.visionBackend = &bb
	}
}

// WithAudioBackend selects an optional audio backend. Pass an empty
// string to leave unset.
func WithAudioBackend(b string) Option {
	return func(c *clientConfig) {
		if b == "" {
			c.audioBackend = nil
			return
		}
		bb := b
		c.audioBackend = &bb
	}
}

// WithMaxTokens caps the engine's total token budget (prompt + output).
// Must be at least the model's smallest prefill signature (typically
// 128). Default is 4096 when unset.
func WithMaxTokens(n int) Option { return func(c *clientConfig) { c.maxTokens = n } }

// WithCacheDir points the engine at a directory it can use for
// artefact caching.
func WithCacheDir(dir string) Option { return func(c *clientConfig) { c.cacheDir = dir } }

// WithActivationDataType selects the activation precision per
// c/engine.h: 0=F32, 1=F16, 2=I16, 3=I8.
func WithActivationDataType(t int) Option {
	return func(c *clientConfig) { c.activationDataType = &t }
}

// WithPrefillChunkSize sets the CPU-backend prefill chunk size for
// dynamic models.
func WithPrefillChunkSize(n int) Option {
	return func(c *clientConfig) { c.prefillChunkSize = &n }
}

// WithSpeculativeDecodingEnabled toggles speculative (multi-token
// prediction) decoding. Requires a model that supports it (e.g.
// Gemma 4).
func WithSpeculativeDecodingEnabled(on bool) Option {
	return func(c *clientConfig) { c.speculativeDecodingEnabled = &on }
}

// WithBenchmarkEnabled turns on benchmark collection. Read the
// captured metrics from a generation result via Response.Benchmark()
// (high-level) or Session.BenchmarkInfo / Conversation.BenchmarkInfo
// (low-level).
func WithBenchmarkEnabled() Option {
	on := true
	return func(c *clientConfig) { c.benchmarkEnabled = &on }
}

// WithParallelFileLoading toggles parallel loading of .litertlm file
// sections. The C side defaults to true; only call this to override.
func WithParallelFileLoading(on bool) Option {
	return func(c *clientConfig) { c.parallelFileLoading = &on }
}

// WithMaxNumImages caps the maximum number of image inputs the engine
// reserves capacity for. Honored only by the legacy engine
// implementation per the upstream C header.
func WithMaxNumImages(n int) Option {
	return func(c *clientConfig) { c.maxNumImages = &n }
}

// WithDispatchLibDir sets the LiteRT dispatch library directory used
// by the NPU backend. Ignored by CPU / GPU backends.
func WithDispatchLibDir(dir string) Option {
	return func(c *clientConfig) { c.dispatchLibDir = dir }
}

// WithLogLevel sets the LiteRT-LM log severity floor. Default is
// LogError when unset (silences INFO/WARN chatter).
func WithLogLevel(level int) Option {
	return func(c *clientConfig) {
		c.logLevel = level
		c.logLevelSet = true
	}
}

// WithDefaultSampler sets the sampler parameters used for every
// Generate call that doesn't override via WithSampler.
func WithDefaultSampler(p SamplerParams) Option {
	return func(c *clientConfig) {
		pp := p
		c.defaultSampler = &pp
	}
}

// ---- per-call options ----

// WithMaxOutputTokens caps output tokens for a single Generate /
// GenerateStream call.
func WithMaxOutputTokens(n int) GenOption {
	return func(c *genConfig) { c.maxOutputTokens = n }
}

// WithSampler overrides the Client's default sampler for a single
// Generate / GenerateStream call.
func WithSampler(p SamplerParams) GenOption {
	return func(c *genConfig) {
		pp := p
		c.sampler = &pp
	}
}

// WithRetries caps the number of retry attempts after a parse failure
// in GenerateData. n=0 means no retries (one total attempt). Retries
// only fire on parse-phase failures; generate-phase errors propagate
// immediately. Ignored by Generate / GenerateStream.
func WithRetries(n int) GenOption {
	return func(c *genConfig) { c.retries = n }
}

// WithSchemaInstruction overrides the default instruction GenerateData
// inserts before the user prompt. The value must be a Printf format
// string with one %s placeholder where the shape hint will be
// rendered. Ignored by Generate / GenerateStream.
func WithSchemaInstruction(s string) GenOption {
	return func(c *genConfig) { c.schemaInstruction = s }
}
