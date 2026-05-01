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
	modelPath     string
	backend       string
	visionBackend *string
	audioBackend  *string

	maxTokens          int
	cacheDir           string
	activationDataType *int
	prefillChunkSize   *int

	// *bool fields stay nil until the user opts in or out, so we can
	// preserve the C-side default when the user doesn't care.
	enableSpeculative   *bool
	enableBenchmark     *bool
	parallelFileLoading *bool

	logLevel    int
	logLevelSet bool

	defaultSampler *SamplerParams
}

// genConfig is the resolved per-call configuration.
type genConfig struct {
	maxOutputTokens int
	sampler         *SamplerParams
}

// ---- constructor options ----

// WithLib sets the directory holding the LiteRT-LM shared libraries.
// Empty string falls back to the LITERTLM_LIB environment variable.
func WithLib(dir string) Option { return func(c *clientConfig) { c.libPath = dir } }

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

// WithEnableSpeculativeDecoding toggles speculative decoding.
func WithEnableSpeculativeDecoding(on bool) Option {
	return func(c *clientConfig) { c.enableSpeculative = &on }
}

// WithEnableBenchmark turns on benchmark collection. Pair with
// Engine().Session().BenchmarkInfo() / similar low-level access to
// retrieve metrics.
func WithEnableBenchmark() Option {
	on := true
	return func(c *clientConfig) { c.enableBenchmark = &on }
}

// WithParallelFileLoading toggles parallel loading of .litertlm file
// sections. The C side defaults to true; only call this to override.
func WithParallelFileLoading(on bool) Option {
	return func(c *clientConfig) { c.parallelFileLoading = &on }
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
