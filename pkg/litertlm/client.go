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
}

// New loads the LiteRT-LM library, applies the supplied Options, and
// constructs an Engine. The returned Client is ready for Generate /
// GenerateStream / NewChat. Close must be called when done.
//
// ctx is reserved for future cancellation of the load step; the
// current C API runs the load synchronously and ignores ctx.
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
	_ = ctx // reserved for future use

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

	if err := Load(cfg.libPath, cfg.backend); err != nil {
		return nil, fmt.Errorf("litertlm: New: %w", err)
	}

	if !cfg.logLevelSet {
		SetMinLogLevel(cfg.logLevel)
	} else {
		SetMinLogLevel(cfg.logLevel)
	}

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
	if cfg.enableSpeculative != nil {
		s.SetEnableSpeculativeDecoding(*cfg.enableSpeculative)
	}
	if cfg.enableBenchmark != nil && *cfg.enableBenchmark {
		s.EnableBenchmark()
	}
	if cfg.parallelFileLoading != nil {
		s.SetParallelFileSectionLoading(*cfg.parallelFileLoading)
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

// openSession creates a fresh single-use Session with the per-call
// genConfig applied. Each Generate call gets its own session because
// the C engine restricts session reuse across prefill/decode cycles.
//
// NewSession is serialised on the Client mutex out of an abundance of
// caution; the C-side thread-safety contract for engine_create_session
// is not documented and contention here is negligible (sessions are
// cheap to construct relative to inference).
func (c *Client) openSession(opts []GenOption) (Session, error) {
	gcfg := genConfig{}
	for _, opt := range opts {
		opt(&gcfg)
	}

	// Resolve effective sampler: per-call > Client default > none.
	sampler := gcfg.sampler
	if sampler == nil {
		sampler = c.cfg.defaultSampler
	}

	var sessCfg SessionConfig
	if gcfg.maxOutputTokens > 0 || sampler != nil {
		var err error
		sessCfg, err = NewSessionConfig()
		if err != nil {
			return 0, err
		}
		if gcfg.maxOutputTokens > 0 {
			sessCfg.SetMaxOutputTokens(gcfg.maxOutputTokens)
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
