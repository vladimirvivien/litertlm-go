// Package loader resolves and loads the LiteRT-LM native shared library.
//
// The loader is a thin, yzma-parallel helper: it understands the platform-
// specific naming convention for shared libraries and surfaces failures as
// ordinary Go errors.
package loader

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/jupiterrider/ffi"
)

// EnvVar is the environment variable consulted when no explicit path is passed
// to LoadLibrary.
const EnvVar = "LITERTLM_LIB"

// LoadLibrary loads the shared library identified by `lib` from the directory
// `path`. If `path` is empty, the value of the LITERTLM_LIB environment
// variable is used instead. Returns an error if no path is available or the
// library cannot be opened.
//
// `lib` is the short, platform-agnostic library name (e.g. "litertlm_c_cpu"
// or "GemmaModelConstraintProvider") — the correct prefix and extension are
// added by GetLibraryFilename.
func LoadLibrary(path, lib string) (ffi.Lib, error) {
	if path == "" {
		path = os.Getenv(EnvVar)
	}
	if path == "" {
		return ffi.Lib{}, fmt.Errorf("library path not specified and %s env variable not set", EnvVar)
	}
	return ffi.Load(GetLibraryFilename(path, lib))
}

// GetLibraryFilename returns the full path to the platform-specific shared
// library file for the given directory and short library name.
//
//   - Linux/FreeBSD: lib<name>.so
//   - Windows:        <name>.dll
//   - macOS:          lib<name>.dylib
func GetLibraryFilename(path, lib string) string {
	switch runtime.GOOS {
	case "linux", "freebsd":
		return filepath.Join(path, fmt.Sprintf("lib%s.so", lib))
	case "windows":
		return filepath.Join(path, fmt.Sprintf("%s.dll", lib))
	case "darwin":
		return filepath.Join(path, fmt.Sprintf("lib%s.dylib", lib))
	default:
		return filepath.Join(path, lib)
	}
}
