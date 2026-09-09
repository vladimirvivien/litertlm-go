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
	oldStderr := os.Stderr
	r, w, _ := os.Pipe()
	os.Stderr = w

	oldLoadAux := loadAuxLibrary
	oldPrepAux := prepAuxSymbol
	oldLoadMainLibVar := loadMainLibVar
	oldLoadFuncsVar := loadFuncsVar
	t.Cleanup(func() {
		loadAuxLibrary = oldLoadAux
		prepAuxSymbol = oldPrepAux
		loadMainLibVar = oldLoadMainLibVar
		loadFuncsVar = oldLoadFuncsVar
		os.Stderr = oldStderr
	})

	loadAuxLibrary = func(path, name string) (ffi.Lib, error) {
		return ffi.Lib{}, nil
	}

	prepAuxSymbol = func(lib ffi.Lib, symbolName string, ret *ffi.Type, args ...*ffi.Type) (ffi.Fun, error) {
		if symbolName == "LiteRtTopKWebGpuSampler_UpdateConfig" {
			return ffi.Fun{}, fmt.Errorf("symbol not found")
		}
		return ffi.Fun{}, nil
	}

	rt.stateMu.Lock()
	oldRealLoaded := rt.realLibraryLoaded
	oldLoadedOnce := rt.loadedOnce
	oldPath := rt.path
	oldPending := rt.pendingLogLevel
	rt.realLibraryLoaded = false
	rt.loadedOnce = false
	rt.pendingLogLevel = new(LogWarning)
	rt.stateMu.Unlock()
	t.Cleanup(func() {
		rt.stateMu.Lock()
		rt.realLibraryLoaded = oldRealLoaded
		rt.loadedOnce = oldLoadedOnce
		rt.path = oldPath
		rt.pendingLogLevel = oldPending
		rt.stateMu.Unlock()
	})

	loadMainLibVar = func(path, backend, libName string) (ffi.Lib, error) {
		return ffi.Lib{}, nil
	}
	loadFuncsVar = func(lib ffi.Lib) error {
		return nil
	}

	if err := Load("dummy_path", "gpu", "dummy_lib"); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	_ = w.Close()
	var buf bytes.Buffer
	_, _ = io.Copy(&buf, r)
	got := buf.String()

	expectContain := "stale WebGPU sampler library detected"
	if !strings.Contains(got, expectContain) {
		t.Errorf("captured stderr = %q, want it to contain %q", got, expectContain)
	}
}
