package litertlm

import (
	"strings"
	"testing"
)

func TestDefaultSamplerParams(t *testing.T) {
	p := DefaultSamplerParams()
	if p.Type != SamplerGreedy {
		t.Errorf("Type = %d, want SamplerGreedy", p.Type)
	}
	if p.TopK != 1 {
		t.Errorf("TopK = %d, want 1", p.TopK)
	}
	if p.TopP != 1.0 {
		t.Errorf("TopP = %v, want 1.0", p.TopP)
	}
	if p.Temperature != 0.0 {
		t.Errorf("Temperature = %v, want 0.0", p.Temperature)
	}
	if p.Seed != 0 {
		t.Errorf("Seed = %d, want 0", p.Seed)
	}
}

func TestSessionConfig_SetSamplerParams_ZeroHandle(t *testing.T) {
	var cfg SessionConfig
	err := cfg.SetSamplerParams(DefaultSamplerParams())
	if err == nil || !strings.Contains(err.Error(), "invalid session config") {
		t.Errorf("expected invalid session config error, got %v", err)
	}
}
