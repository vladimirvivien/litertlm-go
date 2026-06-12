package litertlm

// NewCppBackendForTest exposes the C++ backend constructor to this
// package's external tests (the seam conformance suite). The caller
// owns the engine until the Backend's Close releases it.
func NewCppBackendForTest(engine Engine, benchmarkEnabled bool) Backend {
	return newCppBackend(engine, benchmarkEnabled)
}
