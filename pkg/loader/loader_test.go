package loader

import (
	"runtime"
	"strings"
	"testing"
)

func TestGetLibraryFilename(t *testing.T) {
	got := GetLibraryFilename("/tmp/dist/lib", "litertlm_c_cpu")

	var want string
	switch runtime.GOOS {
	case "linux", "freebsd":
		want = "/tmp/dist/lib/liblitertlm_c_cpu.so"
	case "darwin":
		want = "/tmp/dist/lib/liblitertlm_c_cpu.dylib"
	case "windows":
		want = "/tmp/dist/lib/litertlm_c_cpu.dll"
	default:
		// Unknown GOOS: fall back to concatenation, no prefix/extension.
		want = "/tmp/dist/lib/litertlm_c_cpu"
	}

	if got != want {
		t.Fatalf("GetLibraryFilename(%q) = %q, want %q", runtime.GOOS, got, want)
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
