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

// LoadLibrary loads the main C-API shared library identified by `lib` from
// the directory `path`. If `path` is empty, the value of the LITERTLM_LIB
// environment variable is used instead. Returns an error if no path is
// available or the library cannot be opened.
//
// `lib` is the short library name (e.g. "litertlm_c_cpu") — the correct
// prefix and extension are added by GetLibraryFilename. For plugin /
// accelerator libraries that always carry a "lib" prefix (Gemma, LiteRt,
// LiteRtWebGpuAccelerator, LiteRtTopKWebGpuSampler, etc.), use
// LoadAuxLibrary instead.
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
// library file for the main C-API library, matching the names produced by
// Bazel's cc_binary(linkshared=1) / cc_shared_library rules:
//
//   - Linux/FreeBSD: lib<name>.so
//   - Windows:        <name>.dll        (Bazel/MSVC convention — no "lib" prefix)
//   - macOS:          lib<name>.dylib
//
// Use GetAuxLibraryFilename for plugin/accelerator DLLs shipped under
// prebuilt/<platform>/, which keep the "lib" prefix on every platform.
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

// GetAuxLibraryFilename returns the full path to a platform-specific
// auxiliary shared library that uses the "lib" prefix on every platform.
// This matches the convention of the prebuilt accelerator/plugin libraries
// shipped under prebuilt/<platform>/ in the LiteRT-LM repository — including
// Windows, where the prebuilt plugins retain the "lib" prefix even though
// our own Bazel cc_binary build artifacts omit it.
//
//   - Linux/FreeBSD: lib<name>.so
//   - Windows:       lib<name>.dll
//   - macOS:         lib<name>.dylib
func GetAuxLibraryFilename(path, lib string) string {
	switch runtime.GOOS {
	case "linux", "freebsd":
		return filepath.Join(path, fmt.Sprintf("lib%s.so", lib))
	case "windows":
		return filepath.Join(path, fmt.Sprintf("lib%s.dll", lib))
	case "darwin":
		return filepath.Join(path, fmt.Sprintf("lib%s.dylib", lib))
	default:
		return filepath.Join(path, lib)
	}
}

// LoadAuxLibrary is the LoadLibrary equivalent for auxiliary libraries — see
// GetAuxLibraryFilename for the naming convention.
func LoadAuxLibrary(path, lib string) (ffi.Lib, error) {
	if path == "" {
		path = os.Getenv(EnvVar)
	}
	if path == "" {
		return ffi.Lib{}, fmt.Errorf("library path not specified and %s env variable not set", EnvVar)
	}
	return ffi.Load(GetAuxLibraryFilename(path, lib))
}
