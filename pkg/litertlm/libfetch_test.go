package litertlm_test

import (
	"strings"
	"testing"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func TestLibFetch_InvalidPlatform(t *testing.T) {
	_, err := litertlm.LibFetch("invalid_os", "invalid_arch", "v0.16.0")
	if err == nil {
		t.Fatal("expected error for invalid OS/arch, got nil")
	}
	if !strings.Contains(err.Error(), "unsupported OS/arch") {
		t.Errorf("unexpected error message: %v", err)
	}
}
