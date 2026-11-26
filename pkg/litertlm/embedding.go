package litertlm

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"sync"
	"unsafe"

	"github.com/vladimirvivien/litertlm-go/internal/cstr"
)

// NewEmbeddingEngineSettings constructs an EmbeddingEngineSettings handle.
// vision and audio may be nil to let LiteRT-LM detect them from model assets.
// Caller must Delete() the returned handle.
func NewEmbeddingEngineSettings(modelPath, backend string, vision, audio *string) (EmbeddingEngineSettings, error) {
	if modelPath == "" {
		return 0, fmt.Errorf("litertlm: embedding_engine_settings_create: model path required")
	}
	pathPtr, err := cstr.BytePtrFromString(modelPath)
	if err != nil {
		return 0, err
	}
	backendPtr, visionPtr, audioPtr, err := marshalBackendPtrs(backend, vision, audio)
	if err != nil {
		return 0, err
	}

	s, err := callStatusHandle[EmbeddingEngineSettings](
		embeddingEngineSettingsCreateFunc,
		"embedding_engine_settings_create",
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

// Delete releases an EmbeddingEngineSettings handle.
func (s EmbeddingEngineSettings) Delete() {
	deleteHandle(s, embeddingEngineSettingsDeleteFunc)
}

// SetNumThreads sets the CPU thread count for text/main embedding execution.
func (s EmbeddingEngineSettings) SetNumThreads(n int) error {
	return setHandleScalar(s, "embedding engine settings", embeddingEngineSettingsSetNumThreadsFunc,
		"embedding_engine_settings_set_num_threads", int32(n))
}

// SetAudioNumThreads sets the CPU thread count for audio embedding execution.
func (s EmbeddingEngineSettings) SetAudioNumThreads(n int) error {
	return setHandleScalar(s, "embedding engine settings", embeddingEngineSettingsSetAudioNumThreadsFunc,
		"embedding_engine_settings_set_audio_num_threads", int32(n))
}

// SetCacheDir sets the cache directory across main, vision, and audio executors.
func (s EmbeddingEngineSettings) SetCacheDir(dir string) error {
	return setHandleString(s, "embedding engine settings", embeddingEngineSettingsSetCacheDirFunc,
		"embedding_engine_settings_set_cache_dir", dir)
}

// SetLitertDispatchLibDir sets the LiteRT dispatch library directory for NPU backends.
func (s EmbeddingEngineSettings) SetLitertDispatchLibDir(dir string) error {
	return setHandleString(s, "embedding engine settings", embeddingEngineSettingsSetLitertDispatchLibDirFunc,
		"embedding_engine_settings_set_litert_dispatch_lib_dir", dir)
}

// SetVisionLitertDispatchLibDir sets the LiteRT dispatch library directory for vision NPU backends.
func (s EmbeddingEngineSettings) SetVisionLitertDispatchLibDir(dir string) error {
	return setHandleString(s, "embedding engine settings", embeddingEngineSettingsSetVisionLitertDispatchLibDirFunc,
		"embedding_engine_settings_set_vision_litert_dispatch_lib_dir", dir)
}

// SetAudioLitertDispatchLibDir sets the LiteRT dispatch library directory for audio NPU backends.
func (s EmbeddingEngineSettings) SetAudioLitertDispatchLibDir(dir string) error {
	return setHandleString(s, "embedding engine settings", embeddingEngineSettingsSetAudioLitertDispatchLibDirFunc,
		"embedding_engine_settings_set_audio_litert_dispatch_lib_dir", dir)
}

// SetMaxInputLength sets the maximum sequence length for text encoder signatures.
// Non-positive values unset the option.
func (s EmbeddingEngineSettings) SetMaxInputLength(n int) error {
	return setHandleScalar(s, "embedding engine settings", embeddingEngineSettingsSetMaxInputLengthFunc,
		"embedding_engine_settings_set_max_input_length", int32(n))
}

// SetMinInputLength sets the minimum sequence length for text encoder signatures.
// Negative values unset the option.
func (s EmbeddingEngineSettings) SetMinInputLength(n int) error {
	return setHandleScalar(s, "embedding engine settings", embeddingEngineSettingsSetMinInputLengthFunc,
		"embedding_engine_settings_set_min_input_length", int32(n))
}

// SetVisionTokensPerImage sets the target number of vision tokens generated per image.
// Non-positive values unset the option.
func (s EmbeddingEngineSettings) SetVisionTokensPerImage(n int) error {
	return setHandleScalar(s, "embedding engine settings", embeddingEngineSettingsSetVisionTokensPerImageFunc,
		"embedding_engine_settings_set_vision_tokens_per_image", int32(n))
}

// SetActivationDataType selects the activation precision (0=F32, 1=F16, 2=I16, 3=I8).
func (s EmbeddingEngineSettings) SetActivationDataType(t int) error {
	return setHandleScalar(s, "embedding engine settings", embeddingEngineSettingsSetActivationDataTypeFunc,
		"embedding_engine_settings_set_activation_data_type", int32(t))
}

// ---- EmbeddingOptions ----

// NewEmbeddingOptions creates an EmbeddingOptions handle with C defaults
// (normalize=true, insert_special_tokens=true). Caller must Delete().
func NewEmbeddingOptions() (EmbeddingOptions, error) {
	return callStatusHandle[EmbeddingOptions](embeddingOptionsCreateFunc, "embedding_options_create")
}

// Delete releases an EmbeddingOptions handle.
func (o EmbeddingOptions) Delete() { deleteHandle(o, embeddingOptionsDeleteFunc) }

// SetNormalize configures L2 normalization on output embeddings.
func (o EmbeddingOptions) SetNormalize(on bool) error {
	return setHandleBool(o, "embedding options", embeddingOptionsSetNormalizeFunc,
		"embedding_options_set_normalize", on)
}

// Normalize reports whether output normalization is enabled.
func (o EmbeddingOptions) Normalize() (bool, error) {
	return getHandleBoolErr(o, "embedding options", embeddingOptionsGetNormalizeFunc,
		"embedding_options_get_normalize")
}

// SetInsertSpecialTokens configures automatic BOS/EOS and modality marker insertion.
func (o EmbeddingOptions) SetInsertSpecialTokens(on bool) error {
	return setHandleBool(o, "embedding options", embeddingOptionsSetInsertSpecialTokensFunc,
		"embedding_options_set_insert_special_tokens", on)
}

// InsertSpecialTokens reports whether automatic special-token insertion is enabled.
func (o EmbeddingOptions) InsertSpecialTokens() (bool, error) {
	return getHandleBoolErr(o, "embedding options", embeddingOptionsGetInsertSpecialTokensFunc,
		"embedding_options_get_insert_special_tokens")
}

// SetOverflowStrategy sets the handling strategy when input length exceeds the
// longest supported signature.
func (o EmbeddingOptions) SetOverflowStrategy(strategy OverflowStrategy) error {
	return setHandleScalar(o, "embedding options", embeddingOptionsSetInputOverflowStrategyFunc,
		"embedding_options_set_input_overflow_strategy", int32(strategy))
}

// OverflowStrategy returns the configured overflow strategy.
func (o EmbeddingOptions) OverflowStrategy() (OverflowStrategy, error) {
	v, err := getHandleScalarErr[EmbeddingOptions, int32](o, "embedding options",
		embeddingOptionsGetInputOverflowStrategyFunc, "embedding_options_get_input_overflow_strategy")
	if err != nil {
		return OverflowChunkAndAverage, err
	}
	return OverflowStrategy(v), nil
}

// SetOutputSize configures Matryoshka embedding truncation size. Passing n <= 0
// unsets the override to use the model's full embedding dimension.
func (o EmbeddingOptions) SetOutputSize(n int) error {
	return setHandleScalar(o, "embedding options", embeddingOptionsSetOutputSizeFunc,
		"embedding_options_set_output_size", int32(n))
}

// OutputSize returns the configured truncated output dimension, or ErrNotFound
// when unset.
func (o EmbeddingOptions) OutputSize() (int, error) {
	v, err := getHandleScalarErr[EmbeddingOptions, int32](o, "embedding options",
		embeddingOptionsGetOutputSizeFunc, "embedding_options_get_output_size")
	if err != nil {
		return 0, err
	}
	return int(v), nil
}

// SetVisionTokensPerImage overrides the number of vision tokens per image for
// this request. Passing n <= 0 unsets the override.
func (o EmbeddingOptions) SetVisionTokensPerImage(n int) error {
	return setHandleScalar(o, "embedding options", embeddingOptionsSetVisionTokensPerImageFunc,
		"embedding_options_set_vision_tokens_per_image", int32(n))
}

// VisionTokensPerImage returns the configured per-request vision token budget,
// or ErrNotFound when unset.
func (o EmbeddingOptions) VisionTokensPerImage() (int, error) {
	v, err := getHandleScalarErr[EmbeddingOptions, int32](o, "embedding options",
		embeddingOptionsGetVisionTokensPerImageFunc, "embedding_options_get_vision_tokens_per_image")
	if err != nil {
		return 0, err
	}
	return int(v), nil
}

// ---- EmbeddingResponse & EmbeddingResponses ----

// Delete releases an owned EmbeddingResponse handle. Do not call Delete on a
// borrowed handle returned by EmbeddingResponses.At.
func (r EmbeddingResponse) Delete() { deleteHandle(r, embeddingResponseDeleteFunc) }

// Size returns the number of float elements in the embedding vector.
func (r EmbeddingResponse) Size() (int, error) {
	sz, err := getHandleScalarErr[EmbeddingResponse, uint64](r, "embedding response",
		embeddingResponseGetSizeFunc, "embedding_response_get_size")
	if err != nil {
		return 0, err
	}
	return int(sz), nil
}

// Values returns a Go-owned copy of the embedding float32 slice.
func (r EmbeddingResponse) Values() ([]float32, error) {
	n, err := r.Size()
	if err != nil {
		return nil, err
	}
	ptr, err := getHandleScalarErr[EmbeddingResponse, *float32](r, "embedding response",
		embeddingResponseGetValuesFunc, "embedding_response_get_values")
	if err != nil {
		return nil, err
	}
	return copyCSlice(ptr, uint64(n)), nil
}

// Delete releases an EmbeddingResponses batch collection.
func (rs EmbeddingResponses) Delete() { deleteHandle(rs, embeddingResponsesDeleteFunc) }

// Size returns the number of responses in the batch collection.
func (rs EmbeddingResponses) Size() (int, error) {
	sz, err := getHandleScalarErr[EmbeddingResponses, uint64](rs, "embedding responses",
		embeddingResponsesGetSizeFunc, "embedding_responses_get_size")
	if err != nil {
		return 0, err
	}
	return int(sz), nil
}

// At returns a borrowed handle to the embedding response at index.
// The returned handle is owned by rs and remains valid until rs.Delete();
// callers must not call Delete() on it.
func (rs EmbeddingResponses) At(index int) (EmbeddingResponse, error) {
	if rs == 0 {
		return 0, fmt.Errorf("litertlm: embedding_responses_get_at: invalid embedding responses")
	}
	if index < 0 {
		return 0, fmt.Errorf("litertlm: embedding_responses_get_at: negative index %d", index)
	}
	idx := uint64(index)
	return callStatusHandle[EmbeddingResponse](embeddingResponsesGetAtFunc, "embedding_responses_get_at",
		unsafe.Pointer(&rs), unsafe.Pointer(&idx))
}

// Values copies every embedding vector in the batch into a Go slice of slices.
func (rs EmbeddingResponses) Values() ([][]float32, error) {
	n, err := rs.Size()
	if err != nil {
		return nil, err
	}
	out := make([][]float32, n)
	for i := range n {
		item, err := rs.At(i)
		if err != nil {
			return nil, err
		}
		vals, err := item.Values()
		if err != nil {
			return nil, err
		}
		out[i] = vals
	}
	return out, nil
}

// ---- EmbeddingEngine ----

// NewEmbeddingEngine initializes an EmbeddingEngine from settings. The caller
// retains ownership of settings and should Delete() it once the engine is created.
func NewEmbeddingEngine(settings EmbeddingEngineSettings) (EmbeddingEngine, error) {
	if settings == 0 {
		return 0, fmt.Errorf("litertlm: embedding_engine_create: invalid embedding engine settings")
	}
	return callStatusHandle[EmbeddingEngine](embeddingEngineCreateFunc,
		"embedding_engine_create", unsafe.Pointer(&settings))
}

// Delete releases an EmbeddingEngine handle.
func (e EmbeddingEngine) Delete() { deleteHandle(e, embeddingEngineDeleteFunc) }

// ComputeEmbedding computes a single embedding vector from inputs.
// Pass EmbeddingOptions(0) to use C-side default options. Caller must Delete()
// the returned EmbeddingResponse.
func (e EmbeddingEngine) ComputeEmbedding(inputs []InputData, opts EmbeddingOptions) (EmbeddingResponse, error) {
	if e == 0 {
		return 0, fmt.Errorf("litertlm: embedding_engine_compute_embedding: invalid embedding engine")
	}
	if len(inputs) == 0 {
		return 0, fmt.Errorf("litertlm: embedding_engine_compute_embedding: no inputs")
	}
	ptrs := handlePtrs(inputs)
	inputsPtr := unsafe.Pointer(&ptrs[0])
	n := uint64(len(inputs))
	r, err := callStatusHandle[EmbeddingResponse](
		embeddingEngineComputeEmbeddingFunc,
		"embedding_engine_compute_embedding",
		unsafe.Pointer(&e),
		unsafe.Pointer(&inputsPtr),
		unsafe.Pointer(&n),
		unsafe.Pointer(&opts),
	)
	runtime.KeepAlive(ptrs)
	return r, err
}

// ComputeEmbeddingBatch computes embeddings for a batch of multimodal requests.
// Pass EmbeddingOptions(0) to use C-side default options. Caller must Delete()
// the returned EmbeddingResponses.
func (e EmbeddingEngine) ComputeEmbeddingBatch(batch [][]InputData, opts EmbeddingOptions) (EmbeddingResponses, error) {
	if e == 0 {
		return 0, fmt.Errorf("litertlm: embedding_engine_compute_embedding_batch: invalid embedding engine")
	}
	if len(batch) == 0 {
		return 0, fmt.Errorf("litertlm: embedding_engine_compute_embedding_batch: empty batch")
	}

	rows := make([][]uintptr, len(batch))
	rowPtrs := make([]*uintptr, len(batch))
	numInputsPerBatch := make([]uint64, len(batch))
	for i, item := range batch {
		if len(item) == 0 {
			return 0, fmt.Errorf("litertlm: embedding_engine_compute_embedding_batch: empty inputs at batch index %d", i)
		}
		ptrs := handlePtrs(item)
		rows[i] = ptrs
		rowPtrs[i] = &ptrs[0]
		numInputsPerBatch[i] = uint64(len(item))
	}

	batchPtr := unsafe.Pointer(&rowPtrs[0])
	numsPtr := unsafe.Pointer(&numInputsPerBatch[0])
	batchSize := uint64(len(batch))

	rs, err := callStatusHandle[EmbeddingResponses](
		embeddingEngineComputeEmbeddingBatchFunc,
		"embedding_engine_compute_embedding_batch",
		unsafe.Pointer(&e),
		unsafe.Pointer(&batchPtr),
		unsafe.Pointer(&numsPtr),
		unsafe.Pointer(&batchSize),
		unsafe.Pointer(&opts),
	)
	runtime.KeepAlive(rows)
	runtime.KeepAlive(rowPtrs)
	runtime.KeepAlive(numInputsPerBatch)
	return rs, err
}

// ---- High-level EmbeddingClient ----

// EmbeddingOption tunes a single Embed* call on EmbeddingClient.
type EmbeddingOption func(*embeddingCallConfig)

type embeddingCallConfig struct {
	normalize            *bool
	insertSpecialTokens  *bool
	overflowStrategy     *OverflowStrategy
	outputSize           *int
	visionTokensPerImage *int
}

// WithNormalize configures L2 normalization on the returned embeddings.
func WithNormalize(on bool) EmbeddingOption {
	return func(c *embeddingCallConfig) { c.normalize = new(on) }
}

// WithInsertSpecialTokens configures whether BOS/EOS and modality tokens are inserted automatically.
func WithInsertSpecialTokens(on bool) EmbeddingOption {
	return func(c *embeddingCallConfig) { c.insertSpecialTokens = new(on) }
}

// WithOverflowStrategy sets the input length overflow strategy for the request.
func WithOverflowStrategy(strategy OverflowStrategy) EmbeddingOption {
	return func(c *embeddingCallConfig) { c.overflowStrategy = new(strategy) }
}

// WithOutputSize sets the Matryoshka output truncation dimension. Passing n <= 0
// uses the model's full embedding dimension.
func WithOutputSize(n int) EmbeddingOption {
	return func(c *embeddingCallConfig) { c.outputSize = new(n) }
}

// WithEmbeddingVisionTokensPerImage overrides vision tokens generated per image
// for a single embedding request.
func WithEmbeddingVisionTokensPerImage(n int) EmbeddingOption {
	return func(c *embeddingCallConfig) { c.visionTokensPerImage = new(n) }
}

func resolveEmbeddingConfig(opts []EmbeddingOption) embeddingCallConfig {
	var cfg embeddingCallConfig
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	return cfg
}

func applyEmbeddingOptions(cfg embeddingCallConfig) (EmbeddingOptions, error) {
	o, err := NewEmbeddingOptions()
	if err != nil {
		return 0, err
	}
	if cfg.normalize != nil {
		if err = o.SetNormalize(*cfg.normalize); err != nil {
			o.Delete()
			return 0, err
		}
	}
	if cfg.insertSpecialTokens != nil {
		if err = o.SetInsertSpecialTokens(*cfg.insertSpecialTokens); err != nil {
			o.Delete()
			return 0, err
		}
	}
	if cfg.overflowStrategy != nil {
		if err = o.SetOverflowStrategy(*cfg.overflowStrategy); err != nil {
			o.Delete()
			return 0, err
		}
	}
	if cfg.outputSize != nil {
		if err = o.SetOutputSize(*cfg.outputSize); err != nil {
			o.Delete()
			return 0, err
		}
	}
	if cfg.visionTokensPerImage != nil {
		if err = o.SetVisionTokensPerImage(*cfg.visionTokensPerImage); err != nil {
			o.Delete()
			return 0, err
		}
	}
	return o, nil
}

func withEmbeddingOptions(opts []EmbeddingOption, fn func(EmbeddingOptions) error) error {
	cfg := resolveEmbeddingConfig(opts)
	if cfg.normalize == nil &&
		cfg.insertSpecialTokens == nil &&
		cfg.overflowStrategy == nil &&
		cfg.outputSize == nil &&
		cfg.visionTokensPerImage == nil {
		return fn(0)
	}
	return withTempHandle(
		func() (EmbeddingOptions, error) { return applyEmbeddingOptions(cfg) },
		func(o EmbeddingOptions) { o.Delete() },
		fn,
	)
}

// EmbeddingClient wraps a loaded LiteRT-LM EmbeddingEngine and its settings for
// standalone text and multimodal embedding generation. Call Close when done.
type EmbeddingClient struct {
	cfg      clientConfig
	settings EmbeddingEngineSettings
	engine   EmbeddingEngine

	mu     sync.Mutex
	closed bool
}

// NewEmbeddingClient loads the LiteRT-LM library, applies opts, and initializes a
// standalone EmbeddingEngine. Call Close when done.
func NewEmbeddingClient(ctx context.Context, opts ...Option) (*EmbeddingClient, error) {
	cfg := clientConfig{
		backend: defaultBackend,
	}
	if v := os.Getenv(envLib); v != "" {
		cfg.libPath = v
	}
	if v := os.Getenv(envModel); v != "" {
		cfg.modelPath = v
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	if cfg.err != nil {
		return nil, fmt.Errorf("litertlm: NewEmbeddingClient: configuration error: %w", cfg.err)
	}
	if cfg.modelFD != nil {
		return nil, fmt.Errorf("litertlm: NewEmbeddingClient: file-descriptor loading is not supported for embedding engines")
	}
	if cfg.modelPath == "" {
		return nil, fmt.Errorf("litertlm: NewEmbeddingClient: model path required")
	}

	return runCancellable(ctx,
		func() (*EmbeddingClient, error) { return buildEmbeddingClient(cfg) },
		func(c *EmbeddingClient) { _ = c.Close() },
	)
}

func buildEmbeddingClient(cfg clientConfig) (*EmbeddingClient, error) {
	cAPIBackend := translateBackend(cfg.backend)
	cAPIVision := translateBackendPtr(cfg.visionBackend)
	cAPIAudio := translateBackendPtr(cfg.audioBackend)

	if err := Load(cfg.libPath, cAPIBackend, cfg.libName); err != nil {
		return nil, fmt.Errorf("litertlm: NewEmbeddingClient: %w", err)
	}

	settings, err := NewEmbeddingEngineSettings(cfg.modelPath, cAPIBackend, cAPIVision, cAPIAudio)
	if err != nil {
		return nil, fmt.Errorf("litertlm: NewEmbeddingClient: %w", err)
	}
	if err = applyEmbeddingSettings(settings, &cfg); err != nil {
		settings.Delete()
		return nil, fmt.Errorf("litertlm: NewEmbeddingClient: %w", err)
	}

	engine, err := NewEmbeddingEngine(settings)
	if err != nil {
		settings.Delete()
		return nil, fmt.Errorf("litertlm: NewEmbeddingClient: %w", err)
	}

	return &EmbeddingClient{
		cfg:      cfg,
		settings: settings,
		engine:   engine,
	}, nil
}

func applyEmbeddingSettings(s EmbeddingEngineSettings, cfg *clientConfig) error {
	if cfg.numThreads != nil {
		if err := s.SetNumThreads(*cfg.numThreads); err != nil {
			return err
		}
	}
	if cfg.audioNumThreads != nil {
		if err := s.SetAudioNumThreads(*cfg.audioNumThreads); err != nil {
			return err
		}
	}
	if cfg.cacheDir != "" {
		if err := s.SetCacheDir(cfg.cacheDir); err != nil {
			return err
		}
	}
	if cfg.dispatchLibDir != "" {
		if err := s.SetLitertDispatchLibDir(cfg.dispatchLibDir); err != nil {
			return err
		}
	}
	if cfg.visionDispatchLibDir != "" {
		if err := s.SetVisionLitertDispatchLibDir(cfg.visionDispatchLibDir); err != nil {
			return err
		}
	}
	if cfg.audioDispatchLibDir != "" {
		if err := s.SetAudioLitertDispatchLibDir(cfg.audioDispatchLibDir); err != nil {
			return err
		}
	}
	if cfg.maxInputLength != nil {
		if err := s.SetMaxInputLength(*cfg.maxInputLength); err != nil {
			return err
		}
	}
	if cfg.minInputLength != nil {
		if err := s.SetMinInputLength(*cfg.minInputLength); err != nil {
			return err
		}
	}
	if cfg.maxVisionTokensPerImage != nil {
		if err := s.SetVisionTokensPerImage(*cfg.maxVisionTokensPerImage); err != nil {
			return err
		}
	}
	if cfg.activationDataType != nil {
		if err := s.SetActivationDataType(*cfg.activationDataType); err != nil {
			return err
		}
	}
	return nil
}

// Close releases the underlying EmbeddingEngine and EmbeddingEngineSettings
// handles. Safe to call multiple times.
func (c *EmbeddingClient) Close() error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	c.engine.Delete()
	c.settings.Delete()
	c.engine = 0
	c.settings = 0
	return nil
}

func (c *EmbeddingClient) openEngine() (EmbeddingEngine, error) {
	if c == nil {
		return 0, fmt.Errorf("litertlm: EmbeddingClient is nil")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return 0, fmt.Errorf("litertlm: EmbeddingClient is closed")
	}
	return c.engine, nil
}

// Engine returns the underlying EmbeddingEngine handle for low-level use,
// or 0 if closed. Do not call Delete on the returned handle.
func (c *EmbeddingClient) Engine() EmbeddingEngine {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.engine
}

// Settings returns the underlying EmbeddingEngineSettings handle, or 0 if closed.
// Do not call Delete on the returned handle.
func (c *EmbeddingClient) Settings() EmbeddingEngineSettings {
	if c == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.settings
}

// Capabilities inspects metadata and capabilities of the client's model file.
func (c *EmbeddingClient) Capabilities() (*ModelCapabilities, error) {
	if c == nil || c.cfg.modelPath == "" {
		return nil, fmt.Errorf("litertlm: capabilities inspection requires a model path")
	}
	return InspectModel(c.cfg.modelPath)
}

// Embed computes a single text embedding vector.
func (c *EmbeddingClient) Embed(ctx context.Context, text string, opts ...EmbeddingOption) ([]float32, error) {
	return c.EmbedMulti(ctx, []Part{Text(text)}, opts...)
}

// EmbedMulti computes a single embedding vector for multimodal inputs.
func (c *EmbeddingClient) EmbedMulti(ctx context.Context, parts []Part, opts ...EmbeddingOption) ([]float32, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	engine, err := c.openEngine()
	if err != nil {
		return nil, err
	}
	inputs, err := partsToInputData(parts)
	if err != nil {
		return nil, err
	}
	defer deleteInputData(inputs)

	var vals []float32
	err = withEmbeddingOptions(opts, func(optHandle EmbeddingOptions) error {
		resp, callErr := engine.ComputeEmbedding(inputs, optHandle)
		if callErr != nil {
			return callErr
		}
		defer resp.Delete()
		vals, callErr = resp.Values()
		return callErr
	})
	if err != nil {
		return nil, err
	}
	return vals, nil
}

// EmbedBatch computes embedding vectors for a batch of text strings.
func (c *EmbeddingClient) EmbedBatch(ctx context.Context, texts []string, opts ...EmbeddingOption) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("litertlm: EmbedBatch: empty batch")
	}
	batch := make([][]Part, len(texts))
	for i, text := range texts {
		batch[i] = []Part{Text(text)}
	}
	return c.EmbedMultiBatch(ctx, batch, opts...)
}

// EmbedMultiBatch computes embedding vectors for a batch of multimodal requests.
func (c *EmbeddingClient) EmbedMultiBatch(ctx context.Context, batch [][]Part, opts ...EmbeddingOption) ([][]float32, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	if len(batch) == 0 {
		return nil, fmt.Errorf("litertlm: EmbedMultiBatch: empty batch")
	}
	engine, err := c.openEngine()
	if err != nil {
		return nil, err
	}

	inputBatch := make([][]InputData, len(batch))
	defer func() {
		for _, row := range inputBatch {
			deleteInputData(row)
		}
	}()
	for i, parts := range batch {
		inputs, convErr := partsToInputData(parts)
		if convErr != nil {
			return nil, convErr
		}
		inputBatch[i] = inputs
	}

	var vecs [][]float32
	err = withEmbeddingOptions(opts, func(optHandle EmbeddingOptions) error {
		rs, callErr := engine.ComputeEmbeddingBatch(inputBatch, optHandle)
		if callErr != nil {
			return callErr
		}
		defer rs.Delete()
		vecs, callErr = rs.Values()
		return callErr
	})
	if err != nil {
		return nil, err
	}
	return vecs, nil
}

func partsToInputData(parts []Part) ([]InputData, error) {
	if len(parts) == 0 {
		return nil, fmt.Errorf("litertlm: embedding input requires at least one part")
	}
	inputs := make([]InputData, 0, len(parts))
	for _, p := range parts {
		var (
			in  InputData
			err error
		)
		switch p.kind {
		case partText:
			in, err = NewTextInputString(p.text)
		case partImage:
			in, err = NewBinaryInput(InputImage, p.data)
		case partAudio:
			in, err = NewBinaryInput(InputAudio, p.data)
		default:
			err = fmt.Errorf("litertlm: unsupported part kind %d", p.kind)
		}
		if err != nil {
			deleteInputData(inputs)
			return nil, err
		}
		inputs = append(inputs, in)
	}
	return inputs, nil
}

func deleteInputData(inputs []InputData) {
	for _, in := range inputs {
		in.Delete()
	}
}
