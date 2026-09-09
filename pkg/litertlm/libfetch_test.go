package litertlm_test

import (
	"context"
	"strings"
	"testing"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func TestFetchLib_InvalidPlatform(t *testing.T) {
	_, err := litertlm.FetchLib(context.Background(), "invalid_os", "invalid_arch", "v0.19.1")
	if err == nil {
		t.Fatal("expected error for invalid OS/arch, got nil")
	}
	if !strings.Contains(err.Error(), "unsupported OS/arch") {
		t.Errorf("unexpected error message: %v", err)
	}
}
