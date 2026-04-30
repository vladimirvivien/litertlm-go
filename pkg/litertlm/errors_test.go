package litertlm

import (
	"errors"
	"testing"
)

func TestLoadError(t *testing.T) {
	inner := errors.New("symbol not found")
	got := loadError("litert_lm_engine_create", inner)

	want := `could not load "litert_lm_engine_create": symbol not found`
	if got.Error() != want {
		t.Errorf("Error() = %q, want %q", got.Error(), want)
	}

	// loadError must wrap the original error so callers can inspect via
	// errors.Is / errors.Unwrap.
	if !errors.Is(got, inner) {
		t.Errorf("errors.Is(got, inner) = false, want true (loadError dropped wrap)")
	}
}
