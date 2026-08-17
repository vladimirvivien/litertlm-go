package loader

import (
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
)

func TestGetLibraryFilename(t *testing.T) {
	dir := filepath.Join("tmp", "dist", "lib")
	got := GetLibraryFilename(dir, "litertlm_c_cpu")

	var want string
	switch runtime.GOOS {
	case "linux", "freebsd":
		want = filepath.Join(dir, "liblitertlm_c_cpu.so")
	case "darwin":
		want = filepath.Join(dir, "liblitertlm_c_cpu.dylib")
	case "windows":
		want = filepath.Join(dir, "litertlm_c_cpu.dll")
	default:
		// Unknown GOOS: fall back to concatenation, no prefix/extension.
		want = filepath.Join(dir, "litertlm_c_cpu")
	}

	if got != want {
		t.Fatalf("GetLibraryFilename(%q) = %q, want %q", runtime.GOOS, got, want)
	}
}

func TestGetAuxLibraryFilename(t *testing.T) {
	dir := filepath.Join("tmp", "dist", "lib")
	got := GetAuxLibraryFilename(dir, "GemmaModelConstraintProvider")

	var want string
	switch runtime.GOOS {
	case "linux", "freebsd":
		want = filepath.Join(dir, "libGemmaModelConstraintProvider.so")
	case "darwin":
		want = filepath.Join(dir, "libGemmaModelConstraintProvider.dylib")
	case "windows":
		// Aux libs (prebuilt plugins) keep the "lib" prefix on Windows,
		// unlike the main C-API DLL.
		want = filepath.Join(dir, "libGemmaModelConstraintProvider.dll")
	default:
		want = filepath.Join(dir, "GemmaModelConstraintProvider")
	}

	if got != want {
		t.Fatalf("GetAuxLibraryFilename(%q) = %q, want %q", runtime.GOOS, got, want)
	}
}

func TestLoadLibraryNoPath(t *testing.T) {
	t.Setenv("LITERTLM_LIB", "")
	_, err := LoadLibrary("", "litertlm_c_cpu")
	if err == nil {
		t.Fatal("expected error when path is unset and env unset")
	}
	if !strings.Contains(err.Error(), "LITERTLM_LIB") {
		t.Fatalf("error should mention env var, got %v", err)
	}
}

// Spaces in paths must survive the filename composition. ffi.Load will
// fail because the file doesn't exist, but the error must reference the
// path including the spaces — proving they weren't dropped or split.
func TestGetLibraryFilenameWithSpaces(t *testing.T) {
	dir := filepath.Join("path with spaces", "lib")
	got := GetLibraryFilename(dir, "litertlm_c_cpu")
	if !strings.Contains(got, "path with spaces") {
		t.Errorf("got %q, expected substring %q", got, "path with spaces")
	}
}

func TestGetAuxLibraryFilenameWithSpaces(t *testing.T) {
	dir := filepath.Join("my libs", "v1")
	got := GetAuxLibraryFilename(dir, "GemmaModelConstraintProvider")
	if !strings.Contains(got, "my libs") {
		t.Errorf("got %q, expected substring %q", got, "my libs")
	}
}

// LoadAuxLibrary against a directory that exists but doesn't contain
// the expected lib should surface a load error rather than a panic.
// Uses t.TempDir() so the directory exists, and a name we won't ship.
func TestLoadAuxLibraryMissing(t *testing.T) {
	dir := t.TempDir()
	_, err := LoadAuxLibrary(dir, "DefinitelyNotARealLibrary")
	if err == nil {
		t.Fatal("expected error loading nonexistent aux library")
	}
}

// Same for the main library loader.
func TestLoadLibraryMissing(t *testing.T) {
	dir := t.TempDir()
	_, err := LoadLibrary(dir, "definitely_not_a_real_library")
	if err == nil {
		t.Fatal("expected error loading nonexistent main library")
	}
}

// Env-var fallback must fire when path is empty.
func TestLoadLibraryEnvFallback(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("LITERTLM_LIB", dir)
	// Library still doesn't exist, but the error should be a load error
	// (not the "LITERTLM_LIB unset" sentinel) — proving the fallback fired.
	_, err := LoadLibrary("", "definitely_not_a_real_library")
	if err == nil {
		t.Fatal("expected load error")
	}
	if strings.Contains(err.Error(), "LITERTLM_LIB env variable not set") {
		t.Errorf("expected env fallback to fire; got sentinel error: %v", err)
	}
}

func TestLoader_ConcurrentCalls(t *testing.T) {
	dir := t.TempDir()
	var wg sync.WaitGroup
	for range 20 {
		wg.Add(2)
		go func() {
			defer wg.Done()
			_, _ = LoadLibrary(dir, "concurrent_missing_lib")
		}()
		go func() {
			defer wg.Done()
			_, _ = LoadAuxLibrary(dir, "concurrent_missing_aux")
		}()
	}
	wg.Wait()
}
