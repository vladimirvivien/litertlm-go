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
	"sync"

	"github.com/jupiterrider/ffi"
)

// EnvVar is the environment variable consulted when no explicit path is passed
// to LoadLibrary.
const EnvVar = "LITERTLM_LIB"

var loaderMu sync.Mutex

// LoadLibrary loads the main C-API shared library identified by `lib` from
// the directory `path`. If `path` is empty, the value of the LITERTLM_LIB
// environment variable is used instead. Returns an error if no path is
// available or the library cannot be opened.
//
// `lib` is the short library name (e.g. "litert-lm") — the correct
// prefix and extension are added by LibraryFilename. For legacy
// plugin / accelerator libraries that always carry a "lib" prefix (Gemma,
// LiteRt, LiteRtWebGpuAccelerator, LiteRtTopKWebGpuSampler, etc.), use
// LoadAuxLibrary instead.
func LoadLibrary(path, lib string) (ffi.Lib, error) {
	return loadWith(path, lib, LibraryFilename)
}

// LoadAuxLibrary is the LoadLibrary equivalent for auxiliary libraries — see
// AuxLibraryFilename for the naming convention.
func LoadAuxLibrary(path, lib string) (ffi.Lib, error) {
	return loadWith(path, lib, AuxLibraryFilename)
}

func loadWith(path, lib string, nameFn func(string, string) string) (ffi.Lib, error) {
	loaderMu.Lock()
	defer loaderMu.Unlock()

	if path == "" {
		path = os.Getenv(EnvVar)
	}
	if path == "" {
		return ffi.Lib{}, fmt.Errorf("library path not specified and %s env variable not set", EnvVar)
	}
	setDllDir(path)
	return ffi.Load(nameFn(path, lib))
}

// LibraryFilename returns the full path to the platform-specific shared
// library file for the main C-API library, matching the names produced by
// Bazel's cc_binary(linkshared=1) / cc_shared_library rules:
//
//   - Linux/FreeBSD: lib<name>.so
//   - Windows:       <name>.dll        (Bazel/MSVC convention — no "lib" prefix)
//   - macOS:         lib<name>.dylib
//
// Use AuxLibraryFilename for plugin/accelerator DLLs shipped under
// prebuilt/<platform>/, which keep the "lib" prefix on every platform.
func LibraryFilename(path, lib string) string {
	return libraryFilename(path, lib, "")
}

// AuxLibraryFilename returns the full path to a platform-specific
// auxiliary shared library that uses the "lib" prefix on every platform.
// This matches the convention of the prebuilt accelerator/plugin libraries
// shipped under prebuilt/<platform>/ in the LiteRT-LM repository — including
// Windows, where the prebuilt plugins retain the "lib" prefix even though
// the project's Bazel cc_binary build artifacts omit it.
//
//   - Linux/FreeBSD: lib<name>.so
//   - Windows:       lib<name>.dll
//   - macOS:         lib<name>.dylib
func AuxLibraryFilename(path, lib string) string {
	return libraryFilename(path, lib, "lib")
}

func libraryFilename(path, lib, winPrefix string) string {
	switch runtime.GOOS {
	case "linux", "freebsd":
		return filepath.Join(path, fmt.Sprintf("lib%s.so", lib))
	case "windows":
		return filepath.Join(path, fmt.Sprintf("%s%s.dll", winPrefix, lib))
	case "darwin":
		return filepath.Join(path, fmt.Sprintf("lib%s.dylib", lib))
	default:
		return filepath.Join(path, lib)
	}
}
