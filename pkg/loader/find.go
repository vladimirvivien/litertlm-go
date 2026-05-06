package loader

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// Find returns the first directory in candidates where
// GetLibraryFilename(dir, libName) refers to an existing file.
// candidates are checked in order; the empty string is skipped. On
// miss, the returned error names every resolved filename that was
// checked, so callers see exactly where Find looked.
//
// Find is a pure search helper — it does not consult the LITERTLM_LIB
// environment variable or platform defaults. Callers compose the
// candidate list themselves (typically: explicit path, env var,
// DefaultPaths()).
func Find(libName string, candidates []string) (string, error) {
	if libName == "" {
		return "", fmt.Errorf("loader: Find: libName is required")
	}
	tried := make([]string, 0, len(candidates))
	for _, dir := range candidates {
		if dir == "" {
			continue
		}
		expanded := expandUser(dir)
		full := GetLibraryFilename(expanded, libName)
		if _, err := os.Stat(full); err == nil {
			return expanded, nil
		}
		tried = append(tried, full)
	}
	if len(tried) == 0 {
		return "", fmt.Errorf("loader: Find: no candidate directories supplied for %q", libName)
	}
	return "", fmt.Errorf("loader: %q not found; checked: %s",
		libName, strings.Join(tried, ", "))
}

// DefaultPaths returns the platform-specific list of directories Find
// should consider when neither an explicit path nor LITERTLM_LIB is
// supplied. Order matters: more-specific or user-scoped paths come
// first.
func DefaultPaths() []string {
	switch runtime.GOOS {
	case "windows":
		var paths []string
		if v := os.Getenv("LOCALAPPDATA"); v != "" {
			paths = append(paths, filepath.Join(v, "litertlm", "lib"))
		}
		if v := os.Getenv("PROGRAMFILES"); v != "" {
			paths = append(paths, filepath.Join(v, "litertlm", "lib"))
		}
		return paths
	default:
		paths := []string{}
		if v := os.Getenv("XDG_DATA_HOME"); v != "" {
			paths = append(paths, filepath.Join(v, "litertlm", "lib"))
		}
		paths = append(paths,
			"~/.litertlm/lib",
			"/opt/litertlm/lib",
		)
		if runtime.GOOS == "darwin" {
			paths = append(paths, "/opt/homebrew/lib")
		}
		paths = append(paths, "/usr/local/lib")
		return paths
	}
}

// expandUser expands a leading "~/" to the current user's home
// directory. Returns the input unchanged when expansion is not
// possible.
func expandUser(p string) string {
	if !strings.HasPrefix(p, "~/") {
		return p
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return p
	}
	return filepath.Join(home, p[2:])
}
