package litertlm

import (
	"fmt"
	"strings"
	"testing"
)

// TestOption_Defaults verifies the unconfigured config keeps zero
// values for everything except the fields New() initialises before
// applying user options.
func TestOption_Defaults(t *testing.T) {
	cfg := clientConfig{}
	for _, opt := range []Option{} {
		opt(&cfg)
	}
	if cfg.libPath != "" {
		t.Errorf("libPath = %q, want empty", cfg.libPath)
	}
	if cfg.modelPath != "" {
		t.Errorf("modelPath = %q, want empty", cfg.modelPath)
	}
	if cfg.activationDataType != nil {
		t.Errorf("activationDataType should default to nil")
	}
	if cfg.speculativeDecodingEnabled != nil {
		t.Errorf("speculativeDecodingEnabled should default to nil")
	}
}

// TestOption_LibAndModel exercises the simple string Options.
func TestOption_LibAndModel(t *testing.T) {
	cfg := clientConfig{}
	WithLib("/tmp/lib")(&cfg)
	WithModel("/tmp/model.litertlm")(&cfg)
	if cfg.libPath != "/tmp/lib" {
		t.Errorf("libPath = %q", cfg.libPath)
	}
	if cfg.modelPath != "/tmp/model.litertlm" {
		t.Errorf("modelPath = %q", cfg.modelPath)
	}
}

// TestOption_DispatchLibDir threads WithDispatchLibDir through.
func TestOption_DispatchLibDir(t *testing.T) {
	cfg := clientConfig{}
	if cfg.dispatchLibDir != "" {
		t.Fatalf("dispatchLibDir should default to empty")
	}
	WithDispatchLibDir("/opt/litert-dispatch")(&cfg)
	if cfg.dispatchLibDir != "/opt/litert-dispatch" {
		t.Errorf("dispatchLibDir = %q", cfg.dispatchLibDir)
	}
}

// TestOption_MaxImages threads WithMaxImages through.
func TestOption_MaxImages(t *testing.T) {
	cfg := clientConfig{}
	if cfg.maxImages != nil {
		t.Fatalf("maxImages should default to nil")
	}
	WithMaxImages(4)(&cfg)
	if cfg.maxImages == nil || *cfg.maxImages != 4 {
		t.Errorf("maxImages = %v, want 4", cfg.maxImages)
	}
}

// TestOption_LibName verifies WithLibName threads through to the
// resolved config and respects last-write-wins.
func TestOption_LibName(t *testing.T) {
	cfg := clientConfig{}
	if cfg.libName != "" {
		t.Fatalf("libName should start empty")
	}
	WithLibName("litertlm_c_arm64")(&cfg)
	if cfg.libName != "litertlm_c_arm64" {
		t.Errorf("libName = %q", cfg.libName)
	}
	WithLibName("")(&cfg)
	if cfg.libName != "" {
		t.Errorf("empty WithLibName should clear; got %q", cfg.libName)
	}
}

// TestOption_LastWriteWins ensures composing the same option twice
// keeps the last value (per standard functional-options semantics).
func TestOption_LastWriteWins(t *testing.T) {
	cfg := clientConfig{}
	WithBackend("cpu")(&cfg)
	WithBackend("gpu")(&cfg)
	if cfg.backend != "gpu" {
		t.Errorf("backend = %q, want gpu", cfg.backend)
	}
}

// TestOption_OptionalBackends verifies the *string fields toggle
// between unset and set correctly.
func TestOption_OptionalBackends(t *testing.T) {
	cfg := clientConfig{}
	if cfg.visionBackend != nil {
		t.Fatalf("visionBackend should start nil")
	}
	WithVisionBackend("vision-cpu")(&cfg)
	if cfg.visionBackend == nil || *cfg.visionBackend != "vision-cpu" {
		t.Errorf("visionBackend = %v", cfg.visionBackend)
	}
	WithVisionBackend("")(&cfg)
	if cfg.visionBackend != nil {
		t.Errorf("empty string should clear visionBackend; got %v", cfg.visionBackend)
	}
}

// TestOption_Toggles confirms the bool-toggle options leave the
// config field nil when unset, and set true/false correctly when
// invoked.
func TestOption_Toggles(t *testing.T) {
	cfg := clientConfig{}

	if cfg.speculativeDecodingEnabled != nil || cfg.parallelSectionLoading != nil || cfg.benchmarkEnabled != nil {
		t.Fatal("toggles should default to nil")
	}

	WithSpeculativeDecodingEnabled(true)(&cfg)
	if cfg.speculativeDecodingEnabled == nil || !*cfg.speculativeDecodingEnabled {
		t.Errorf("speculativeDecodingEnabled = %v", cfg.speculativeDecodingEnabled)
	}

	WithParallelSectionLoading(false)(&cfg)
	if cfg.parallelSectionLoading == nil || *cfg.parallelSectionLoading {
		t.Errorf("parallelSectionLoading = %v", cfg.parallelSectionLoading)
	}

	WithBenchmarkEnabled()(&cfg)
	if cfg.benchmarkEnabled == nil || !*cfg.benchmarkEnabled {
		t.Errorf("benchmarkEnabled = %v", cfg.benchmarkEnabled)
	}
}

// TestOption_DefaultSampler captures by value, so later mutations to
// the caller's struct don't bleed into the Client.
func TestOption_DefaultSampler(t *testing.T) {
	cfg := clientConfig{}
	p := SamplerParams{Type: SamplerGreedy, TopK: 1, Temperature: 0.5}
	WithDefaultSampler(p)(&cfg)
	if cfg.defaultSampler == nil {
		t.Fatal("defaultSampler should be populated")
	}
	if cfg.defaultSampler.Temperature != 0.5 {
		t.Errorf("Temperature = %v", cfg.defaultSampler.Temperature)
	}
	// Mutate caller's struct; client copy must be unaffected.
	p.Temperature = 1.0
	if cfg.defaultSampler.Temperature != 0.5 {
		t.Errorf("expected client copy isolated from caller mutation")
	}
}

// TestRuntimeOption_Composition checks per-call options.
func TestRuntimeOption_Composition(t *testing.T) {
	g := runtimeConfig{}
	WithMaxOutputTokens(256)(&g)
	if g.maxOutputTokens != 256 {
		t.Errorf("maxOutputTokens = %d", g.maxOutputTokens)
	}
	WithSampler(SamplerParams{Type: SamplerTopP, TopP: 0.9})(&g)
	if g.sampler == nil || g.sampler.TopP != 0.9 {
		t.Errorf("sampler = %v", g.sampler)
	}
	WithVisualTokenBudget(512)(&g)
	if g.visualTokenBudget == nil || *g.visualTokenBudget != 512 {
		t.Errorf("visualTokenBudget = %v", g.visualTokenBudget)
	}
}

// TestBuildOptionalArgs_UnsetReturnsZero verifies that with no Chat-side
// knobs set, no C handle is materialized.
func TestBuildOptionalArgs_UnsetReturnsZero(t *testing.T) {
	opts, err := buildOptionalArgs(runtimeConfig{})
	if err != nil {
		t.Fatalf("buildOptionalArgs: %v", err)
	}
	if opts != 0 {
		t.Errorf("opts = %v, want OptionalArgs(0) when no knobs set", opts)
	}
}

func TestBuildOptionalArgs_WithMaxOutputTokens(t *testing.T) {
	if !realLibraryLoaded {
		t.Skip("skipping test requiring real library")
	}
	cfg := runtimeConfig{maxOutputTokens: 128}
	opts, err := buildOptionalArgs(cfg)
	if err != nil {
		t.Fatalf("buildOptionalArgs: %v", err)
	}
	defer opts.Delete()
	if opts == 0 {
		t.Fatal("expected non-zero OptionalArgs when maxOutputTokens is set")
	}
}

func TestBuildOptionalArgs_WithBudgetAndMaxTokens(t *testing.T) {
	if !realLibraryLoaded {
		t.Skip("skipping test requiring real library")
	}
	vtb := 256
	cfg := runtimeConfig{
		visualTokenBudget: &vtb,
		maxOutputTokens:   128,
	}
	opts, err := buildOptionalArgs(cfg)
	if err != nil {
		t.Fatalf("buildOptionalArgs: %v", err)
	}
	defer opts.Delete()
	if opts == 0 {
		t.Fatal("expected non-zero OptionalArgs when visualTokenBudget and maxOutputTokens are set")
	}
}

// TestDiagnoseLoadError verifies that diagnoseLoadError correctly wraps
// engine preparation failures but leaves other errors as standard New errors.
func TestDiagnoseLoadError(t *testing.T) {
	tests := []struct {
		name          string
		err           error
		backend       string
		expectContain string
	}{
		{
			name:          "failed to prepare error on GPU",
			err:           fmt.Errorf("some internal error: failed to prepare model"),
			backend:       "gpu",
			expectContain: "GPU memory allocation issue, such as the 2 GB per-allocation limit",
		},
		{
			name:          "failed to prepare error on CPU",
			err:           fmt.Errorf("some internal error: failed to prepare runner"),
			backend:       "cpu",
			expectContain: "GPU memory allocation issue, such as the 2 GB per-allocation limit",
		},
		{
			name:          "generic access denied is unmodified",
			err:           fmt.Errorf("Access is denied"),
			backend:       "gpu",
			expectContain: "litertlm: New: Access is denied",
		},
		{
			name:          "unrelated error",
			err:           fmt.Errorf("file not found"),
			backend:       "cpu",
			expectContain: "litertlm: New: file not found",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := clientConfig{backend: tc.backend}
			got := diagnoseLoadError(tc.err, cfg)
			if got == nil {
				t.Fatal("expected non-nil error")
			}
			if !strings.Contains(got.Error(), tc.expectContain) {
				t.Errorf("diagnoseLoadError(%q) = %q, want it to contain %q", tc.err.Error(), got.Error(), tc.expectContain)
			}
		})
	}
}

// TestOption_NewV014Options threads threading and LoRA configurations.
func TestOption_NewV014Options(t *testing.T) {
	cfg := clientConfig{}

	if cfg.modelFd != nil || cfg.numThreads != nil || cfg.audioNumThreads != nil || cfg.loraRank != nil || cfg.audioLoraRank != nil {
		t.Fatal("new options should default to nil")
	}

	WithModelFd(10)(&cfg)
	if cfg.modelFd == nil || *cfg.modelFd != 10 {
		t.Errorf("modelFd = %v, want 10", cfg.modelFd)
	}

	WithNumThreads(4)(&cfg)
	if cfg.numThreads == nil || *cfg.numThreads != 4 {
		t.Errorf("numThreads = %v, want 4", cfg.numThreads)
	}

	WithAudioNumThreads(2)(&cfg)
	if cfg.audioNumThreads == nil || *cfg.audioNumThreads != 2 {
		t.Errorf("audioNumThreads = %v, want 2", cfg.audioNumThreads)
	}

	WithLoRARank(8)(&cfg)
	if cfg.loraRank == nil || *cfg.loraRank != 8 {
		t.Errorf("loraRank = %v, want 8", cfg.loraRank)
	}

	WithSupportedLoRARanks([]int{8, 16})(&cfg)
	if len(cfg.supportedLoraRanks) != 2 || cfg.supportedLoraRanks[0] != 8 || cfg.supportedLoraRanks[1] != 16 {
		t.Errorf("supportedLoraRanks = %v, want [8, 16]", cfg.supportedLoraRanks)
	}

	WithAudioLoRARank(4)(&cfg)
	if cfg.audioLoraRank == nil || *cfg.audioLoraRank != 4 {
		t.Errorf("audioLoraRank = %v, want 4", cfg.audioLoraRank)
	}

	WithSupportedAudioLoRARanks([]int{4, 8})(&cfg)
	if len(cfg.supportedAudioLoraRanks) != 2 || cfg.supportedAudioLoraRanks[0] != 4 || cfg.supportedAudioLoraRanks[1] != 8 {
		t.Errorf("supportedAudioLoraRanks = %v, want [4, 8]", cfg.supportedAudioLoraRanks)
	}
}
