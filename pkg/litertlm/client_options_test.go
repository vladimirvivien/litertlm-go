package litertlm

import (
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

// TestRuntimeArgsFrom_UnsetReturnsZero verifies that with no Chat-side
// knobs set, the resolved args are the zero value (engine defaults).
func TestRuntimeArgsFrom_UnsetReturnsZero(t *testing.T) {
	args := runtimeArgsFrom(runtimeConfig{})
	if args != (RuntimeArgs{}) {
		t.Errorf("args = %+v, want zero RuntimeArgs when no knobs set", args)
	}
}
