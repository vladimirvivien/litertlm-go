package litertlm

// Option configures a Client at New time.
type Option func(*clientConfig)

// RuntimeOption tunes a single inference call without rebuilding the
// Client. Apply via the variadic opts parameter on Client.Generate*,
// GenerateData / GenerateDataMulti, and Chat.Send*.
type RuntimeOption func(*runtimeConfig)

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
	parallelSectionLoading     *bool

	dispatchLibDir string

	maxImages *int

	defaultSampler *SamplerParams

	modelFd                 *int
	numThreads              *int
	audioNumThreads         *int
	loraRank                *int
	supportedLoraRanks      []int
	audioLoraRank           *int
	supportedAudioLoraRanks []int

	err error
}

// runtimeConfig is the resolved per-call configuration.
type runtimeConfig struct {
	maxOutputTokens int
	sampler         *SamplerParams

	// GenerateData-specific. Generate ignores these fields.
	retries           int
	schemaInstruction string

	// Chat.Send*-specific. Generate / GenerateData ignore this.
	// Nil-when-unset preserves the C-side default.
	visualTokenBudget *int

	// Chat.Send / SendMulti / SendToolResult only. When true, the
	// dispatch loop is bypassed and the first reply containing
	// tool calls is returned to the caller for manual handling.
	// Ignored by Client.Generate*, SendStream, and SendMultiStream.
	returnToolRequests bool

	// Chat.Send* only. <=1 dispatches tool calls sequentially (the
	// default); >1 dispatches in parallel capped at that many
	// in-flight handlers. Ignored by Client.Generate* and
	// GenerateData.
	maxConcurrentTools int
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

// WithParallelSectionLoading toggles parallel deserialization of
// sections within the .litertlm container (weights, tokenizer,
// decode and prefill graphs, multimodal adapters). The C side
// defaults to true; only call this to override.
func WithParallelSectionLoading(on bool) Option {
	return func(c *clientConfig) { c.parallelSectionLoading = &on }
}

// WithDispatchLibDir sets the LiteRT dispatch library directory used
// by the NPU backend. Ignored by CPU / GPU backends.
func WithDispatchLibDir(dir string) Option {
	return func(c *clientConfig) { c.dispatchLibDir = dir }
}

// WithMaxImages caps the number of image inputs per generation. The
// C side gates this option on the legacy ARTISAN backends
// (cpu_artisan / gpu_artisan); calls with backend "cpu" or "gpu" are
// accepted but have no effect.
func WithMaxImages(n int) Option {
	return func(c *clientConfig) { c.maxImages = &n }
}

// WithDefaultSampler sets the sampler parameters used for every
// Generate call that doesn't override via WithSampler.
func WithDefaultSampler(p SamplerParams) Option {
	return func(c *clientConfig) {
		pp := p
		c.defaultSampler = &pp
	}
}

// WithModelFd sets the raw file descriptor of the model file to load.
// The engine takes ownership of this file descriptor and closes it when done.
func WithModelFd(fd int) Option {
	return func(c *clientConfig) { c.modelFd = &fd }
}

// WithNumThreads sets the number of threads for the CPU backend.
func WithNumThreads(n int) Option {
	return func(c *clientConfig) { c.numThreads = &n }
}

// WithAudioNumThreads sets the number of threads for the audio CPU backend.
func WithAudioNumThreads(n int) Option {
	return func(c *clientConfig) { c.audioNumThreads = &n }
}

// WithLoRARank sets the LoRA rank for the engine.
func WithLoRARank(rank int) Option {
	return func(c *clientConfig) { c.loraRank = &rank }
}

// WithSupportedLoRARanks sets the supported LoRA ranks for the engine.
func WithSupportedLoRARanks(ranks []int) Option {
	return func(c *clientConfig) { c.supportedLoraRanks = ranks }
}

// WithAudioLoRARank sets the audio LoRA rank for the engine.
func WithAudioLoRARank(rank int) Option {
	return func(c *clientConfig) { c.audioLoraRank = &rank }
}

// WithSupportedAudioLoRARanks sets the supported audio LoRA ranks for the engine.
func WithSupportedAudioLoRARanks(ranks []int) Option {
	return func(c *clientConfig) { c.supportedAudioLoraRanks = ranks }
}

// ---- per-call options ----

// WithMaxOutputTokens caps output tokens for a single Generate /
// GenerateStream call.
func WithMaxOutputTokens(n int) RuntimeOption {
	return func(c *runtimeConfig) { c.maxOutputTokens = n }
}

// WithSampler overrides the Client's default sampler for a single
// Generate / GenerateStream call.
func WithSampler(p SamplerParams) RuntimeOption {
	return func(c *runtimeConfig) {
		pp := p
		c.sampler = &pp
	}
}

// WithRetries caps the number of retry attempts after a parse failure
// in GenerateData. n=0 means no retries (one total attempt). Retries
// only fire on parse-phase failures; generate-phase errors propagate
// immediately. Ignored by Generate / GenerateStream.
func WithRetries(n int) RuntimeOption {
	return func(c *runtimeConfig) { c.retries = n }
}

// WithSchemaInstruction overrides the default instruction GenerateData
// inserts before the user prompt. The value must be a Printf format
// string with one %s placeholder where the shape hint will be
// rendered. Ignored by Generate / GenerateStream.
func WithSchemaInstruction(s string) RuntimeOption {
	return func(c *runtimeConfig) { c.schemaInstruction = s }
}

// WithVisualTokenBudget caps the number of vision tokens consumed by
// a single Chat.SendMulti / SendMultiStream turn. Effective on Gemma 4
// vision-enabled models; text-only and audio turns ignore it.
// Ignored by Client.Generate* and GenerateData.
func WithVisualTokenBudget(n int) RuntimeOption {
	return func(c *runtimeConfig) { c.visualTokenBudget = &n }
}

// WithReturnToolRequests bypasses the Chat dispatch loop for a single
// turn. When true, the first reply containing tool calls is returned
// directly via Reply.ToolCalls() for manual handling, even when every
// call maps to a registered ManagedTool. Pair with Chat.SendToolResult
// to feed the result back into the conversation.
//
// Applies to Chat.Send, SendMulti, and SendToolResult. Ignored by
// Client.Generate*, GenerateData, SendStream, and SendMultiStream.
func WithReturnToolRequests(on bool) RuntimeOption {
	return func(c *runtimeConfig) { c.returnToolRequests = on }
}

// WithMaxConcurrentTools caps the number of tool handlers Chat dispatch
// runs concurrently when a single reply contains multiple tool calls.
// n <= 1 keeps the default sequential dispatch; n > 1 enables parallel
// dispatch capped at n in-flight handlers. Tool result ordering in the
// follow-up tool-role message matches the model's original call order
// regardless of completion order.
//
// Handlers must be safe to invoke from multiple goroutines when n > 1.
// Applies to Chat.Send*. Ignored by Client.Generate* and GenerateData.
func WithMaxConcurrentTools(n int) RuntimeOption {
	return func(c *runtimeConfig) { c.maxConcurrentTools = n }
}
