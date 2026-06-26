package litertlm

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/jupiterrider/ffi"
)

func TestLoad_WebGpuSamplerStaleWarning(t *testing.T) {
	// Temporarily redirect stderr to capture the logged warning
	oldStderr := os.Stderr
	r, w, _ := os.Pipe()
	os.Stderr = w

	// Swap loadAuxLibrary and prepAuxSymbol with mocks
	oldLoadAux := loadAuxLibrary
	oldPrepAux := prepAuxSymbol
	defer func() {
		loadAuxLibrary = oldLoadAux
		prepAuxSymbol = oldPrepAux
		os.Stderr = oldStderr
	}()

	// Mock loadAuxLibrary to succeed for LiteRtTopKWebGpuSampler and return a zero-value ffi.Lib
	loadAuxLibrary = func(path, name string) (ffi.Lib, error) {
		if name == "LiteRtTopKWebGpuSampler" {
			return ffi.Lib{}, nil
		}
		// Return dummy handle for other libs
		return ffi.Lib{}, nil
	}

	// Mock prepAuxSymbol to fail specifically for LiteRtTopKWebGpuSampler_UpdateConfig
	prepAuxSymbol = func(lib ffi.Lib, symbolName string, ret *ffi.Type, args ...*ffi.Type) (ffi.Fun, error) {
		if symbolName == "LiteRtTopKWebGpuSampler_UpdateConfig" {
			return ffi.Fun{}, fmt.Errorf("symbol not found")
		}
		return ffi.Fun{}, nil
	}

	// Mock other dependencies of Load so it doesn't fail early on mainLib or loadFuncs
	oldLoadMainLibVar := loadMainLibVar
	oldLoadFuncsVar := loadFuncsVar
	defer func() {
		loadMainLibVar = oldLoadMainLibVar
		loadFuncsVar = oldLoadFuncsVar
	}()

	loadMainLibVar = func(path, backend, libName string) (ffi.Lib, error) {
		return ffi.Lib{}, nil
	}
	loadFuncsVar = func(lib ffi.Lib) error {
		return nil
	}

	// Reset loadedOnce so Load runs
	loadedOnce = false
	oldPending := pendingLogLevel
	lvl := LogWarning
	pendingLogLevel = &lvl
	defer func() {
		loadedOnce = true
		pendingLogLevel = oldPending
	}()

	// Call Load
	err := Load("dummy_path", "gpu", "dummy_lib")
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	// Close the write pipe and read the captured output
	_ = w.Close()
	var buf bytes.Buffer
	_, _ = io.Copy(&buf, r)
	got := buf.String()

	// Verify that the warning was logged
	expectContain := "stale WebGPU sampler library detected"
	if !strings.Contains(got, expectContain) {
		t.Errorf("captured stderr = %q, want it to contain %q", got, expectContain)
	}
}
