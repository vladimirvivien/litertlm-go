package litertlm_test

import (
	"testing"

	"github.com/vladimirvivien/litertlm-go/backend/backendtest"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

// TestCppBackend_Conformance runs the seam conformance suite against
// the C++ backend. Same environment as the other integration tests
// (LITERTLM_TEST_LIB + LITERTLM_TEST_MODEL, skips in -short).
func TestCppBackend_Conformance(t *testing.T) {
	backendtest.Run(t, func(tb testing.TB) litertlm.Backend {
		tt, ok := tb.(*testing.T)
		if !ok {
			tb.Fatal("conformance constructor requires *testing.T")
		}
		libDir, modelPath := requireTestModel(tt)

		if err := litertlm.Load(libDir, "cpu", ""); err != nil {
			tt.Fatalf("Load: %v", err)
		}
		litertlm.SetMinLogLevel(litertlm.LogQuiet)

		settings, err := litertlm.NewEngineSettings(modelPath, "cpu", nil, nil)
		if err != nil {
			tt.Fatalf("NewEngineSettings: %v", err)
		}
		tt.Cleanup(func() { settings.Delete() })

		engine, err := litertlm.NewEngine(settings)
		if err != nil {
			tt.Fatalf("NewEngine: %v", err)
		}
		// The Backend owns the engine; its Close releases it.
		return litertlm.NewCppBackendForTest(engine, false)
	})
}
