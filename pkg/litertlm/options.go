package litertlm

import "slices"

// Option configures a Client at New time.
type Option func(*clientConfig)

// RuntimeOption tunes a single inference call without rebuilding the
// Client. Apply via the variadic opts parameter on Client.Generate*,
// GenerateData / GenerateDataMulti, and Chat.Send*.
type RuntimeOption func(*runtimeConfig)

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

	speculativeDecodingEnabled *bool
	speculativeDecoding        *SpeculativeDecodingConfig
	benchmarkEnabled           *bool
	parallelSectionLoading     *bool
	enableYNNPACK              *bool
	maxVisionTokensPerImage    *int
	metalResidencySet          *bool
	gpuDecodeStepsPerSync      *int
	gpuWaitForWeightUploads    *bool
	useRingbuffersLocalAttn    *bool
	singleThreadedExecution    *bool
	selectedSignatures         []string

	dispatchLibDir string

	maxImages *int

	defaultSampler *SamplerParams

	modelFD                 *int
	numThreads              *int
	audioNumThreads         *int
	loraRank                *int
	supportedLoraRanks      []int
	audioLoraRank           *int
	supportedAudioLoraRanks []int

	err error
}

type runtimeConfig struct {
	maxOutputTokens int
	sampler         *SamplerParams

	retries           int
	schemaInstruction string

	visualTokenBudget *int

	returnToolRequests bool
	maxConcurrentTools int

	thinking          *ThinkingConfig
	constraintType    ConstraintType
	constraintPattern string
	toolChoice        ToolChoice
	repetitionPenalty *RepetitionPenaltyConfig
	noRepeatNgram     *NoRepeatNgramConfig
	suppressTokens    *SuppressTokensConfig
}

// WithLib sets the directory holding the LiteRT-LM shared libraries.
// Empty string falls back to LITERTLM_LIB, then loader.DefaultPaths.
func WithLib(dir string) Option { return func(c *clientConfig) { c.libPath = dir } }

// WithLibName overrides the main C-API library short-name (default "litert-lm",
// with legacy backend-specific fallback). Empty string falls back to LITERTLM_LIB_NAME.
func WithLibName(name string) Option { return func(c *clientConfig) { c.libName = name } }

// WithModel sets the path to the .litertlm model file. Empty string
// falls back to LITERTLM_MODEL.
func WithModel(path string) Option { return func(c *clientConfig) { c.modelPath = path } }

// WithBackend selects the inference backend ("cpu" or "gpu"). Default is "cpu".
func WithBackend(b string) Option { return func(c *clientConfig) { c.backend = b } }

// WithVisionBackend selects an optional vision backend. Empty leaves unset.
func WithVisionBackend(b string) Option {
	return func(c *clientConfig) {
		if b == "" {
			c.visionBackend = nil
			return
		}
		c.visionBackend = new(b)
	}
}

// WithAudioBackend selects an optional audio backend. Empty leaves unset.
func WithAudioBackend(b string) Option {
	return func(c *clientConfig) {
		if b == "" {
			c.audioBackend = nil
			return
		}
		c.audioBackend = new(b)
	}
}

// WithMaxTokens caps the engine's total token budget (prompt + output).
func WithMaxTokens(n int) Option { return func(c *clientConfig) { c.maxTokens = n } }

// WithCacheDir points the engine at a directory for artefact caching.
func WithCacheDir(dir string) Option { return func(c *clientConfig) { c.cacheDir = dir } }

// WithActivationDataType selects the activation precision per c/engine.h (0=F32, 1=F16, 2=I16, 3=I8).
func WithActivationDataType(t int) Option {
	return func(c *clientConfig) { c.activationDataType = new(t) }
}

// WithPrefillChunkSize sets the CPU-backend prefill chunk size for dynamic models.
func WithPrefillChunkSize(n int) Option {
	return func(c *clientConfig) { c.prefillChunkSize = new(n) }
}

// WithSpeculativeDecodingEnabled toggles speculative (multi-token prediction) decoding.
func WithSpeculativeDecodingEnabled(on bool) Option {
	return func(c *clientConfig) { c.speculativeDecodingEnabled = new(on) }
}

// WithSpeculativeDecoding configures engine-wide speculative decoding mode and lenience.
func WithSpeculativeDecoding(cfg SpeculativeDecodingConfig) Option {
	return func(c *clientConfig) { c.speculativeDecoding = new(cfg) }
}

// WithYNNPACK toggles YNNPACK CPU delegation before XNNPACK.
func WithYNNPACK(on bool) Option {
	return func(c *clientConfig) { c.enableYNNPACK = new(on) }
}

// WithMaxVisionTokensPerImage caps vision tokens generated per image.
func WithMaxVisionTokensPerImage(n int) Option {
	return func(c *clientConfig) { c.maxVisionTokensPerImage = new(n) }
}

// WithMetalResidencySet toggles Metal residency sets on Apple GPU backends.
func WithMetalResidencySet(on bool) Option {
	return func(c *clientConfig) { c.metalResidencySet = new(on) }
}

// WithGPUDecodeStepsPerSync configures decode steps between GPU syncs.
func WithGPUDecodeStepsPerSync(n int) Option {
	return func(c *clientConfig) { c.gpuDecodeStepsPerSync = new(n) }
}

// WithGPUWaitForWeightUploads configures whether GPU initialization waits for weight uploads.
func WithGPUWaitForWeightUploads(on bool) Option {
	return func(c *clientConfig) { c.gpuWaitForWeightUploads = new(on) }
}

// WithRingbuffersLocalAttention toggles ringbuffer KV cache storage for local attention layers.
func WithRingbuffersLocalAttention(on bool) Option {
	return func(c *clientConfig) { c.useRingbuffersLocalAttn = new(on) }
}

// WithSingleThreadedExecution configures caller-driven single-threaded session execution.
func WithSingleThreadedExecution(on bool) Option {
	return func(c *clientConfig) { c.singleThreadedExecution = new(on) }
}

// WithSelectedSignatures restricts the main executor to the listed model signatures
// (requires LiteRT-LM C API 1.1.0+).
func WithSelectedSignatures(signatures ...string) Option {
	cloned := slices.Clone(signatures)
	return func(c *clientConfig) { c.selectedSignatures = cloned }
}

// WithBenchmarkEnabled turns on benchmark collection.
func WithBenchmarkEnabled() Option {
	return func(c *clientConfig) { c.benchmarkEnabled = new(true) }
}

// WithParallelSectionLoading toggles parallel deserialization of .litertlm sections.
func WithParallelSectionLoading(on bool) Option {
	return func(c *clientConfig) { c.parallelSectionLoading = new(on) }
}

// WithDispatchLibDir sets the LiteRT dispatch library directory for NPU backends.
func WithDispatchLibDir(dir string) Option {
	return func(c *clientConfig) { c.dispatchLibDir = dir }
}

// WithMaxImages caps the number of image inputs per generation for legacy backends.
func WithMaxImages(n int) Option {
	return func(c *clientConfig) { c.maxImages = new(n) }
}

// WithDefaultSampler sets the default sampler parameters for Generate / Chat.
func WithDefaultSampler(p SamplerParams) Option {
	return func(c *clientConfig) { c.defaultSampler = new(p) }
}

// WithModelFD sets the raw file descriptor of the model file to load.
// The engine takes ownership of this file descriptor and closes it when done.
func WithModelFD(fd int) Option {
	return func(c *clientConfig) { c.modelFD = new(fd) }
}

// WithNumThreads sets the number of threads for the CPU backend.
func WithNumThreads(n int) Option {
	return func(c *clientConfig) { c.numThreads = new(n) }
}

// WithAudioNumThreads sets the number of threads for the audio CPU backend.
func WithAudioNumThreads(n int) Option {
	return func(c *clientConfig) { c.audioNumThreads = new(n) }
}

// WithLoRARank sets the LoRA rank for the engine.
func WithLoRARank(rank int) Option {
	return func(c *clientConfig) { c.loraRank = new(rank) }
}

// WithSupportedLoRARanks sets the supported LoRA ranks for the engine.
func WithSupportedLoRARanks(ranks []int) Option {
	cloned := slices.Clone(ranks)
	return func(c *clientConfig) { c.supportedLoraRanks = cloned }
}

// WithAudioLoRARank sets the audio LoRA rank for the engine.
func WithAudioLoRARank(rank int) Option {
	return func(c *clientConfig) { c.audioLoraRank = new(rank) }
}

// WithSupportedAudioLoRARanks sets the supported audio LoRA ranks for the engine.
func WithSupportedAudioLoRARanks(ranks []int) Option {
	cloned := slices.Clone(ranks)
	return func(c *clientConfig) { c.supportedAudioLoraRanks = cloned }
}

// WithMaxOutputTokens caps output tokens for a single generation turn.
func WithMaxOutputTokens(n int) RuntimeOption {
	return func(c *runtimeConfig) { c.maxOutputTokens = n }
}

// WithSampler overrides the Client's default sampler for a single generation call.
func WithSampler(p SamplerParams) RuntimeOption {
	return func(c *runtimeConfig) { c.sampler = new(p) }
}

// WithRetries caps the number of retry attempts after a parse failure in GenerateData.
func WithRetries(n int) RuntimeOption {
	return func(c *runtimeConfig) { c.retries = n }
}

// WithSchemaInstruction overrides the default schema instruction in GenerateData.
func WithSchemaInstruction(s string) RuntimeOption {
	return func(c *runtimeConfig) { c.schemaInstruction = s }
}

// WithVisualTokenBudget caps the vision tokens consumed by a single multimodal Chat turn.
func WithVisualTokenBudget(n int) RuntimeOption {
	return func(c *runtimeConfig) { c.visualTokenBudget = new(n) }
}

// WithReturnToolRequests bypasses the Chat dispatch loop for a single turn.
func WithReturnToolRequests(on bool) RuntimeOption {
	return func(c *runtimeConfig) { c.returnToolRequests = on }
}

// WithMaxConcurrentTools caps concurrent tool handler dispatch per reply.
func WithMaxConcurrentTools(n int) RuntimeOption {
	return func(c *runtimeConfig) { c.maxConcurrentTools = n }
}

// WithTurnThinking overrides the conversation thinking config for a single turn.
func WithTurnThinking(cfg ThinkingConfig) RuntimeOption {
	return func(c *runtimeConfig) { c.thinking = new(cfg) }
}

// WithConstraint applies a grammar constraint (regex or JSON schema) for a single turn.
func WithConstraint(typ ConstraintType, pattern string) RuntimeOption {
	return func(c *runtimeConfig) {
		c.constraintType = typ
		c.constraintPattern = pattern
	}
}

// WithToolChoice configures per-turn tool choice policy under constrained decoding
// (requires LiteRT-LM C API 1.1.0+).
func WithToolChoice(choice ToolChoice) RuntimeOption {
	return func(c *runtimeConfig) { c.toolChoice = choice }
}

// WithRepetitionPenalty applies multiplicative/subtractive repetition penalties for a single turn.
func WithRepetitionPenalty(cfg RepetitionPenaltyConfig) RuntimeOption {
	return func(c *runtimeConfig) { c.repetitionPenalty = new(cfg) }
}

// WithNoRepeatNgram bans repeating token n-grams for a single turn.
func WithNoRepeatNgram(cfg NoRepeatNgramConfig) RuntimeOption {
	return func(c *runtimeConfig) { c.noRepeatNgram = new(cfg) }
}

// WithSuppressTokens bans the specified token IDs from being sampled for a single turn.
func WithSuppressTokens(tokenIDs ...int) RuntimeOption {
	cloned := slices.Clone(tokenIDs)
	return func(c *runtimeConfig) {
		c.suppressTokens = &SuppressTokensConfig{TokenIDs: cloned}
	}
}
