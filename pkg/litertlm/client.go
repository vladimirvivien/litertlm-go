package litertlm

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
)

// Default values applied when no Option overrides them.
const (
	defaultBackend   = "cpu"
	defaultMaxTokens = 4096
)

// Environment variable names consulted by New when WithLib / WithModel
// are not supplied (or are passed empty).
const (
	envLib   = "LITERTLM_LIB"
	envModel = "LITERTLM_MODEL"
)

// Client is a high-level handle that wraps a loaded library, an
// EngineSettings, and an Engine. Reuse a single Client for the life of
// the process; create one per process per model.
type Client struct {
	cfg      clientConfig
	settings EngineSettings
	engine   Engine

	mu     sync.Mutex
	closed bool
	tools  map[string]ToolDefinition // populated by RegisterTool
}

// New loads the LiteRT-LM library, applies the supplied Options, and
// constructs an Engine. The returned Client is ready for Generate /
// GenerateStream / NewChat. Close must be called when done.
//
// Cancelling ctx during New returns ctx.Err() to the caller promptly.
// The underlying C work (library load, model file mmap, engine
// construction) has no cancel hook and runs to completion in the
// background; if it eventually succeeds after cancellation, the
// resulting handles are released automatically so nothing leaks.
//
// New uses these defaults:
//   - backend: "cpu"
//   - max tokens: 4096
//   - lib path: WithLib value, else $LITERTLM_LIB
//   - model path: WithModel value, else $LITERTLM_MODEL
//
// New does not touch the LiteRT-LM log severity floor. The C side
// defaults to LogInfo (verbose). Call SetMinLogLevel(LogQuiet)
// before New to silence the loader and executor init chatter.
//
// For finer-grained control, the low-level constructors (Load,
// NewEngineSettings, NewEngine, Engine.NewSession) remain available
// and are what New itself uses internally.
func New(ctx context.Context, opts ...Option) (*Client, error) {
	cfg := clientConfig{
		backend:   defaultBackend,
		maxTokens: defaultMaxTokens,
	}

	// Env-var fallbacks first, user options override.
	if v := os.Getenv(envLib); v != "" {
		cfg.libPath = v
	}
	if v := os.Getenv(envModel); v != "" {
		cfg.modelPath = v
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	if cfg.err != nil {
		return nil, fmt.Errorf("litertlm: New: configuration error: %w", cfg.err)
	}

	if cfg.modelPath == "" && cfg.modelFd == nil {
		return nil, fmt.Errorf("litertlm: New: model path or raw file descriptor required")
	}

	return runCancellable(ctx,
		func() (*Client, error) { return buildClient(cfg) },
		func(c *Client) { _ = c.Close() },
	)
}

// buildClient performs the synchronous C-side work of constructing a
// Client. Split out so New can run it under runCancellable.
func buildClient(cfg clientConfig) (*Client, error) {
	// Translate Go-side "apple" / "metal" backend alias to "gpu" for C-API loading and settings initialization.
	cApiBackend := translateBackend(cfg.backend)
	cApiVision := translateBackendPtr(cfg.visionBackend)
	cApiAudio := translateBackendPtr(cfg.audioBackend)

	if err := Load(cfg.libPath, cApiBackend, cfg.libName); err != nil {
		return nil, fmt.Errorf("litertlm: New: %w", err)
	}

	var settings EngineSettings
	var err error
	if cfg.modelFd != nil {
		settings, err = NewEngineSettingsFromFd(*cfg.modelFd, cApiBackend, cApiVision, cApiAudio)
	} else {
		settings, err = NewEngineSettings(cfg.modelPath, cApiBackend, cApiVision, cApiAudio)
	}
	if err != nil {
		return nil, fmt.Errorf("litertlm: New: %w", err)
	}
	err = applySettings(settings, &cfg)
	if err != nil {
		settings.Delete()
		return nil, fmt.Errorf("litertlm: New: %w", err)
	}

	engine, err := NewEngine(settings)
	if err != nil {
		settings.Delete()
		return nil, diagnoseLoadError(err, cfg)
	}

	return &Client{
		cfg:      cfg,
		settings: settings,
		engine:   engine,
	}, nil
}

// applySettings funnels the cfg's resolved values through the
// individual EngineSettings setters. Only setters whose corresponding
// option was supplied (or that have a non-zero default) actually run,
// preserving the C-side defaults for everything else.
func applySettings(s EngineSettings, cfg *clientConfig) error {
	if cfg.maxTokens > 0 {
		s.SetMaxNumTokens(cfg.maxTokens)
	}
	if cfg.cacheDir != "" {
		s.SetCacheDir(cfg.cacheDir)
	}
	if cfg.activationDataType != nil {
		s.SetActivationDataType(*cfg.activationDataType)
	}
	if cfg.prefillChunkSize != nil {
		s.SetPrefillChunkSize(*cfg.prefillChunkSize)
	}
	if cfg.speculativeDecodingEnabled != nil {
		s.SetEnableSpeculativeDecoding(*cfg.speculativeDecodingEnabled)
	}
	if cfg.benchmarkEnabled != nil && *cfg.benchmarkEnabled {
		s.EnableBenchmark()
	}
	if cfg.parallelSectionLoading != nil {
		s.SetParallelFileSectionLoading(*cfg.parallelSectionLoading)
	}
	if cfg.dispatchLibDir != "" {
		s.SetLitertDispatchLibDir(cfg.dispatchLibDir)
	}
	if cfg.maxImages != nil {
		s.SetMaxNumImages(*cfg.maxImages)
	}
	if cfg.numThreads != nil {
		s.SetNumThreads(*cfg.numThreads)
	}
	if cfg.audioNumThreads != nil {
		s.SetAudioNumThreads(*cfg.audioNumThreads)
	}
	if cfg.loraRank != nil {
		s.SetLoraRank(*cfg.loraRank)
	}
	if len(cfg.supportedLoraRanks) > 0 {
		if err := s.SetSupportedLoraRanks(cfg.supportedLoraRanks); err != nil {
			return err
		}
	}
	if cfg.audioLoraRank != nil {
		s.SetAudioLoraRank(*cfg.audioLoraRank)
	}
	if len(cfg.supportedAudioLoraRanks) > 0 {
		if err := s.SetSupportedAudioLoraRanks(cfg.supportedAudioLoraRanks); err != nil {
			return err
		}
	}
	return nil
}

// Close releases the engine and engine-settings handles. Safe to call
// multiple times. The shared library stays loaded for the life of the
// process — purego does not expose dlclose, and the C runtime's
// lifetime is tied to the process anyway.
func (c *Client) Close() error {
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

// Engine returns the underlying Engine handle for advanced use cases
// (manual prefill→decode, scoring, token introspection). The handle's
// lifetime is owned by the Client; do not call Delete on it.
func (c *Client) Engine() Engine { return c.engine }

// Settings returns the underlying EngineSettings handle. Same lifetime
// caveat as Engine — do not call Delete.
func (c *Client) Settings() EngineSettings { return c.settings }

// Tokenize returns the token IDs the model's tokenizer produces for
// the given text. Useful for budgeting prompts against the engine's
// max-token limit.
func (c *Client) Tokenize(text string) ([]int32, error) {
	return c.engine.Tokenize(text)
}

// TokenLength is a convenience over Tokenize that returns just the
// token count.
func (c *Client) TokenLength(text string) (int, error) {
	tokens, err := c.engine.Tokenize(text)
	if err != nil {
		return 0, err
	}
	return len(tokens), nil
}

// resolveRuntimeConfig applies opts to a fresh runtimeConfig, returning the
// result. Public entry points apply opts here once and pass the
// resolved cfg into the internal helpers — that lets GenerateData
// reuse the same cfg without re-applying options that don't matter
// to the underlying generation path.
func resolveRuntimeConfig(opts []RuntimeOption) runtimeConfig {
	cfg := runtimeConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// diagnoseLoadError enriches engine preparation failures with diagnostic advice
// for GPU memory allocation limits (such as the 2 GB per-allocation limit).
func diagnoseLoadError(err error, cfg clientConfig) error {
	if err == nil {
		return nil
	}
	errMsg := err.Error()

	// Intercept engine preparation failures which typically indicate allocation/resource issues
	if strings.Contains(errMsg, "failed to prepare") {
		return fmt.Errorf("litertlm: New: %w (this may indicate a GPU memory allocation issue, such as the 2 GB per-allocation limit on the WebGPU/D3D12 delegate, or insufficient VRAM; consider running on CPU or using a smaller model)", err)
	}

	return fmt.Errorf("litertlm: New: %w", err)
}

func translateBackend(b string) string {
	if b == "apple" || b == "metal" {
		return "gpu"
	}
	if b == "ynnpack" {
		return "cpu"
	}
	return b
}

func translateBackendPtr(b *string) *string {
	if b == nil {
		return nil
	}
	val := translateBackend(*b)
	return &val
}
