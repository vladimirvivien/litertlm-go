package litertgo_test

import (
	"testing"

	"github.com/vladimirvivien/litertlm-go/backend/backendtest"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

// TestGoBackend_Conformance runs the seam conformance suite against
// the litert-go backend. Same environment as the integration tests
// (LITERT_LIB + LITERTLM_TEST_MODEL, skips in -short).
func TestGoBackend_Conformance(t *testing.T) {
	backendtest.Run(t, func(tb testing.TB) litertlm.Backend {
		b, _ := openBackend(tb)
		return b
	})
}
