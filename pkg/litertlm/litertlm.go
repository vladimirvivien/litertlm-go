// Package litertlm is a purego-backed, cgo-free Go wrapper around
// Google's LiteRT-LM C API (see c/engine.h in the LiteRT-LM repository).
//
// The flow for local inference is:
//
//	litertlm.Load(libDir, "cpu")                 // once; "" auto-picks
//	defer litertlm.Close()
//
//	s := litertlm.NewEngineSettings(modelPath, "cpu", nil, nil)
//	defer s.Delete()
//	engine, _ := litertlm.NewEngine(s)
//	defer engine.Delete()
//
//	session, _ := engine.NewSession(0)           // 0 = default config
//	defer session.Delete()
//
//	resp, _ := session.GenerateContent([]litertlm.InputData{
//	    litertlm.NewTextInputString("What is the tallest building in the world?"),
//	})
//	defer resp.Delete()
//	fmt.Println(resp.Text(0))
//
// Ownership: every New*/Generate* result must be paired with .Delete().
// No finalizers are attached. Strings returned by accessor methods are
// copied into Go memory, so they remain valid after their parent is
// deleted.
package litertlm

import (
	"sync"

	"github.com/jupiterrider/ffi"

	"github.com/vladimirvivien/litertlm-go/pkg/loader"
)

// libByBackend maps the backend selector to the main C-API shared lib:
// "cpu" → litertlm_c_cpu, "gpu" → litertlm_c. Unknown values fall back
// to GPU first, then CPU.
var libByBackend = map[string]string{
	"cpu": "litertlm_c_cpu",
	"gpu": "litertlm_c",
}

// auxLibs are required by the C API at load time and must sit next to
// the main library.
var auxLibs = []string{
	"GemmaModelConstraintProvider",
}

// optionalLibs are GPU accelerator plugins from LiteRT-LM's prebuilt/
// directory. Their absence is not an error; backend="gpu" calls will
// just fail at runtime if any are missing.
var optionalLibs = []string{
	"LiteRt",
	"LiteRtWebGpuAccelerator",
	"LiteRtTopKWebGpuSampler",
	"LiteRtMetalAccelerator", // macOS only
}

var (
	loadMu     sync.Mutex
	loadedOnce bool
	libPath    string
)

// LibPath returns the directory from which Load opened the shared
// libraries. Empty until a successful Load.
func LibPath() string {
	loadMu.Lock()
	defer loadMu.Unlock()
	return libPath
}

// Load opens the LiteRT-LM shared libraries and binds C entry points.
// path is the directory containing the libs; "" consults LITERTLM_LIB.
// backend selects "cpu" or "gpu"; "" or unknown picks GPU then CPU.
//
// Auxiliary libs are dlopen'd before the main lib so DT_NEEDED references
// resolve without requiring LD_LIBRARY_PATH / DYLD_LIBRARY_PATH. Load is
// safe to call concurrently; subsequent successful calls are no-ops.
func Load(path, backend string) error {
	loadMu.Lock()
	defer loadMu.Unlock()
	if loadedOnce {
		return nil
	}

	for _, name := range optionalLibs {
		_, _ = loader.LoadAuxLibrary(path, name)
	}
	for _, name := range auxLibs {
		if _, err := loader.LoadAuxLibrary(path, name); err != nil {
			return err
		}
	}

	mainLib, err := loadMainLib(path, backend)
	if err != nil {
		return err
	}
	if err := loadFuncs(mainLib); err != nil {
		return err
	}

	libPath = path
	loadedOnce = true
	return nil
}

// loadMainLib opens the C-API library matching backend, falling back
// from GPU to CPU when backend is empty/unknown.
func loadMainLib(path, backend string) (ffi.Lib, error) {
	if short, ok := libByBackend[backend]; ok {
		return loader.LoadLibrary(path, short)
	}
	if lib, err := loader.LoadLibrary(path, "litertlm_c"); err == nil {
		return lib, nil
	}
	return loader.LoadLibrary(path, "litertlm_c_cpu")
}

// Close is a no-op retained for API symmetry: purego doesn't expose
// dlclose, and the C library's lifetime is tied to the process.
func Close() {}
