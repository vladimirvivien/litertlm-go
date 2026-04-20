package litertlm

import "fmt"

// loadError wraps a per-symbol binding failure with the symbol name, matching
// yzma's `loadError` helper so package-level Load() errors are immediately
// actionable ("could not load %q: ...").
func loadError(name string, err error) error {
	return fmt.Errorf("could not load %q: %w", name, err)
}
