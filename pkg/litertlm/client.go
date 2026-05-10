package litertlm

import (
	"context"
	"fmt"
	"os"
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
//   - log level: LogError
//   - lib path: WithLib value, else $LITERTLM_LIB
//   - model path: WithModel value, else $LITERTLM_MODEL
//
// For finer-grained control, the low-level constructors (Load,
// NewEngineSettings, NewEngine, Engine.NewSession) remain available
// and are what New itself uses internally.
func New(ctx context.Context, opts ...Option) (*Client, error) {
	cfg := clientConfig{
		backend:   defaultBackend,
		maxTokens: defaultMaxTokens,
		logLevel:  LogError,
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

	if cfg.modelPath == "" {
		return nil, fmt.Errorf("litertlm: New: model path required (WithModel or %s env)", envModel)
	}

	return runCancellable(ctx,
		func() (*Client, error) { return buildClient(cfg) },
		func(c *Client) { _ = c.Close() },
	)
}

// buildClient performs the synchronous C-side work of constructing a
// Client. Split out so New can run it under runCancellable.
func buildClient(cfg clientConfig) (*Client, error) {
	if err := Load(cfg.libPath, cfg.backend, cfg.libName); err != nil {
		return nil, fmt.Errorf("litertlm: New: %w", err)
	}

	SetMinLogLevel(cfg.logLevel)

	settings, err := NewEngineSettings(cfg.modelPath, cfg.backend, cfg.visionBackend, cfg.audioBackend)
	if err != nil {
		return nil, fmt.Errorf("litertlm: New: %w", err)
	}
	applySettings(settings, &cfg)

	engine, err := NewEngine(settings)
	if err != nil {
		settings.Delete()
		return nil, fmt.Errorf("litertlm: New: %w", err)
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
func applySettings(s EngineSettings, cfg *clientConfig) {
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
	if cfg.parallelFileLoading != nil {
		s.SetParallelFileSectionLoading(*cfg.parallelFileLoading)
	}
	if cfg.maxNumImages != nil {
		s.SetMaxNumImages(*cfg.maxNumImages)
	}
	if cfg.dispatchLibDir != "" {
		s.SetLitertDispatchLibDir(cfg.dispatchLibDir)
	}
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

// resolveGenConfig applies opts to a fresh genConfig, returning the
// result. Public entry points apply opts here once and pass the
// resolved cfg into the internal helpers — that lets GenerateData
// reuse the same cfg without re-applying options that don't matter
// to the underlying generation path.
func resolveGenConfig(opts []GenOption) genConfig {
	cfg := genConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// openSession creates a fresh single-use Session populated from cfg.
// Each generation call gets its own session because the C engine
// restricts session reuse across prefill/decode cycles.
//
// NewSession is serialised on the Client mutex out of an abundance of
// caution; the C-side thread-safety contract for engine_create_session
// is not documented and contention here is negligible (sessions are
// cheap to construct relative to inference).
func (c *Client) openSession(cfg genConfig) (Session, error) {
	// Resolve effective sampler: per-call > Client default > none.
	sampler := cfg.sampler
	if sampler == nil {
		sampler = c.cfg.defaultSampler
	}

	var sessCfg SessionConfig
	if cfg.maxOutputTokens > 0 || sampler != nil {
		var err error
		sessCfg, err = NewSessionConfig()
		if err != nil {
			return 0, err
		}
		if cfg.maxOutputTokens > 0 {
			sessCfg.SetMaxOutputTokens(cfg.maxOutputTokens)
		}
		if sampler != nil {
			sessCfg.SetSamplerParams(*sampler)
		}
	}

	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		sessCfg.Delete()
		return 0, fmt.Errorf("litertlm: Client is closed")
	}
	sess, err := c.engine.NewSession(sessCfg)
	c.mu.Unlock()
	// SessionConfig fields are copied by NewSession; safe to delete now.
	sessCfg.Delete()
	return sess, err
}
