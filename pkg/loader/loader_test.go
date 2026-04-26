package loader

import (
	"path/filepath"
	"runtime"
	"strings"
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
