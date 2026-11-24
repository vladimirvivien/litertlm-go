package litertlm

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
)

const (
	defaultBackend   = "cpu"
	defaultMaxTokens = 4096
)

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
// constructs an Engine. Close must be called when done.
func New(ctx context.Context, opts ...Option) (*Client, error) {
	cfg := clientConfig{
		backend:   defaultBackend,
		maxTokens: defaultMaxTokens,
	}

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

	if cfg.modelPath == "" && cfg.modelFD == nil {
		return nil, fmt.Errorf("litertlm: New: model path or raw file descriptor required")
	}

	return runCancellable(ctx,
		func() (*Client, error) { return buildClient(cfg) },
		func(c *Client) { _ = c.Close() },
	)
}

func buildClient(cfg clientConfig) (*Client, error) {
	cAPIBackend := translateBackend(cfg.backend)
	cAPIVision := translateBackendPtr(cfg.visionBackend)
	cAPIAudio := translateBackendPtr(cfg.audioBackend)

	if err := Load(cfg.libPath, cAPIBackend, cfg.libName); err != nil {
		return nil, fmt.Errorf("litertlm: New: %w", err)
	}

	var settings EngineSettings
	var err error
	if cfg.modelFD != nil {
		settings, err = NewEngineSettingsFromFD(*cfg.modelFD, cAPIBackend, cAPIVision, cAPIAudio)
	} else {
		settings, err = NewEngineSettings(cfg.modelPath, cAPIBackend, cAPIVision, cAPIAudio)
	}
	if err != nil {
		return nil, fmt.Errorf("litertlm: New: %w", err)
	}
	if err = applySettings(settings, &cfg); err != nil {
		settings.Delete()
		return nil, fmt.Errorf("litertlm: New: %w", err)
	}

	engine, err := NewEngine(settings)
	if err != nil {
		settings.Delete()
		return nil, diagnoseLoadError(err)
	}

	return &Client{
		cfg:      cfg,
		settings: settings,
		engine:   engine,
	}, nil
}

func applySettings(s EngineSettings, cfg *clientConfig) error {
	if cfg.maxTokens > 0 {
		if err := s.SetMaxNumTokens(cfg.maxTokens); err != nil {
			return err
		}
	}
	if cfg.cacheDir != "" {
		if err := s.SetCacheDir(cfg.cacheDir); err != nil {
			return err
		}
	}
	if cfg.activationDataType != nil {
		if err := s.SetActivationDataType(*cfg.activationDataType); err != nil {
			return err
		}
	}
	if cfg.prefillChunkSize != nil {
		if err := s.SetPrefillChunkSize(*cfg.prefillChunkSize); err != nil {
			return err
		}
	}
	if cfg.speculativeDecoding != nil {
		if err := s.SetSpeculativeDecodingConfig(*cfg.speculativeDecoding); err != nil {
			return err
		}
	} else if cfg.speculativeDecodingEnabled != nil {
		if err := s.SetEnableSpeculativeDecoding(*cfg.speculativeDecodingEnabled); err != nil {
			return err
		}
	}
	if cfg.enableYNNPACK != nil {
		if err := s.SetEnableYNNPACK(*cfg.enableYNNPACK); err != nil {
			return err
		}
	} else if cfg.backend == "ynnpack" {
		if err := s.SetEnableYNNPACK(true); err != nil {
			return err
		}
	}
	if cfg.maxVisionTokensPerImage != nil {
		if err := s.SetMaxVisionTokensPerImage(*cfg.maxVisionTokensPerImage); err != nil {
			return err
		}
	}
	if cfg.metalResidencySet != nil {
		if err := s.SetGPUEnableMetalResidencySet(*cfg.metalResidencySet); err != nil {
			return err
		}
	}
	if cfg.gpuDecodeStepsPerSync != nil {
		if err := s.SetGPUDecodeStepsPerSync(*cfg.gpuDecodeStepsPerSync); err != nil {
			return err
		}
	}
	if cfg.gpuWaitForWeightUploads != nil {
		if err := s.SetGPUWaitForWeightUploads(*cfg.gpuWaitForWeightUploads); err != nil {
			return err
		}
	}
	if cfg.useRingbuffersLocalAttn != nil {
		if err := s.SetUseRingbuffersLocalAttention(*cfg.useRingbuffersLocalAttn); err != nil {
			return err
		}
	}
	if cfg.singleThreadedExecution != nil {
		if err := s.SetSingleThreadedExecution(*cfg.singleThreadedExecution); err != nil {
			return err
		}
	}
	if len(cfg.selectedSignatures) > 0 {
		if err := s.SetSelectedSignatures(cfg.selectedSignatures); err != nil {
			return err
		}
	}
	if cfg.benchmarkEnabled != nil && *cfg.benchmarkEnabled {
		if err := s.EnableBenchmark(); err != nil {
			return err
		}
	}
	if cfg.parallelSectionLoading != nil {
		if err := s.SetParallelFileSectionLoading(*cfg.parallelSectionLoading); err != nil {
			return err
		}
	}
	if cfg.dispatchLibDir != "" {
		if err := s.SetLitertDispatchLibDir(cfg.dispatchLibDir); err != nil {
			return err
		}
	}
	if cfg.maxImages != nil {
		if err := s.SetMaxNumImages(*cfg.maxImages); err != nil {
			return err
		}
	}
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
	if cfg.loraRank != nil {
		if err := s.SetLoRARank(*cfg.loraRank); err != nil {
			return err
		}
	}
	if len(cfg.supportedLoraRanks) > 0 {
		if err := s.SetSupportedLoRARanks(cfg.supportedLoraRanks); err != nil {
			return err
		}
	}
	if cfg.audioLoraRank != nil {
		if err := s.SetAudioLoRARank(*cfg.audioLoraRank); err != nil {
			return err
		}
	}
	if len(cfg.supportedAudioLoraRanks) > 0 {
		if err := s.SetSupportedAudioLoRARanks(cfg.supportedAudioLoraRanks); err != nil {
			return err
		}
	}
	return nil
}

// Close releases the engine and engine-settings handles. Safe to call multiple times.
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

func (c *Client) openEngine() (Engine, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return 0, fmt.Errorf("litertlm: Client is closed")
	}
	return c.engine, nil
}

// Engine returns the underlying Engine handle for advanced use cases.
// Do not call Delete on it; returns 0 if closed.
func (c *Client) Engine() Engine {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.engine
}

// Settings returns the underlying EngineSettings handle.
// Do not call Delete on it; returns 0 if closed.
func (c *Client) Settings() EngineSettings {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.settings
}

// Tokenize returns the token IDs the model's tokenizer produces for text.
func (c *Client) Tokenize(text string) ([]int32, error) {
	engine, err := c.openEngine()
	if err != nil {
		return nil, err
	}
	return engine.Tokenize(text)
}

// TokenLength returns the token count of text.
func (c *Client) TokenLength(text string) (int, error) {
	tokens, err := c.Tokenize(text)
	if err != nil {
		return 0, err
	}
	return len(tokens), nil
}

// Capabilities queries the model capabilities and metadata of the client's model file.
func (c *Client) Capabilities() (*ModelCapabilities, error) {
	if c.cfg.modelPath == "" {
		return nil, fmt.Errorf("litertlm: capabilities inspection requires a file path model source")
	}
	return InspectModel(c.cfg.modelPath)
}

func resolveRuntimeConfig(opts []RuntimeOption) runtimeConfig {
	cfg := runtimeConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

func diagnoseLoadError(err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, ErrResourceExhausted) || strings.Contains(err.Error(), "failed to prepare") {
		return fmt.Errorf("litertlm: New: %w (this may indicate a GPU memory allocation issue, such as the 2 GB per-allocation limit on the WebGPU/D3D12 delegate, or insufficient VRAM; consider running on CPU or using a smaller model)", err)
	}
	return fmt.Errorf("litertlm: New: %w", err)
}

func (c *Client) singleThreaded() bool {
	return c.cfg.singleThreadedExecution != nil && *c.cfg.singleThreadedExecution
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
	return new(translateBackend(*b))
}
