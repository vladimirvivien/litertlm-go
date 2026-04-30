package litertlm

import (
	"testing"
	"unsafe"
)

// SamplerParams must be 20 bytes (5 × 4-byte fields) to match the C
// struct. A regression here would corrupt the entire SetSamplerParams
// call.
func TestSamplerParamsLayout(t *testing.T) {
	if got, want := unsafe.Sizeof(SamplerParams{}), uintptr(20); got != want {
		t.Fatalf("sizeof(SamplerParams) = %d, want %d", got, want)
	}
}

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
