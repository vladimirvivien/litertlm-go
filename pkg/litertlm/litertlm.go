// Package litertlm is a purego-backed, cgo-free Go wrapper around
// Google's LiteRT-LM C API (see c/engine.h in the LiteRT-LM repository).
//
// The flow for local inference is:
//
//	litertlm.Load(libDir, "cpu", "")             // once; "" auto-picks
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
//	in, _ := litertlm.NewTextInputString("What is the tallest building in the world?")
//	defer in.Delete()
//
//	resp, _ := session.GenerateContent([]litertlm.InputData{in})
//	defer resp.Delete()
//	fmt.Println(resp.Text(0))
//
// Ownership: every New*/Generate* result must be paired with .Delete().
// No finalizers are attached. Strings returned by accessor methods are
// copied into Go memory, so they remain valid after their parent is
// deleted.
package litertlm

import (
	"fmt"
	"os"
	"sync"

	"github.com/jupiterrider/ffi"

	"github.com/vladimirvivien/litertlm-go/pkg/loader"
)

// envLibName is consulted by Load when libName is empty.
const envLibName = "LITERTLM_LIB_NAME"

// libByBackend maps the backend selector to the main C-API shared lib:
// "cpu" → litertlm_c_cpu, "gpu" → litertlm_c. Unknown values fall back
// to GPU first, then CPU.
var libByBackend = map[string]string{
	"cpu":     "litertlm_c_cpu",
	"gpu":     "litertlm_c",
	"apple":   "litertlm_c",
	"metal":   "litertlm_c",
	"ynnpack": "litertlm_c_cpu",
}

// auxLibs are required by the C API at load time and must sit next to
// the main library.
var auxLibs = []string{
	"GemmaModelConstraintProvider",
}

// optionalLibs are GPU accelerator plugins from LiteRT-LM's prebuilt/
// directory. Their absence is not an error; backend="gpu", "apple", or "metal" calls will
// just fail at runtime if any are missing.
var optionalLibs = []string{
	"LiteRt",
	"LiteRtWebGpuAccelerator",
	"LiteRtTopKWebGpuSampler",
	"LiteRtMetalAccelerator", // macOS only
	"LiteRtTopKMetalSampler", // macOS only
}

var (
	loadMu          sync.Mutex
	loadedOnce      bool
	libPath         string
	pendingLogLevel *LogLevel
	loadAuxLibrary  = loader.LoadAuxLibrary
	loadMainLibVar  = loadMainLib
	loadFuncsVar    = loadFuncs
	prepAuxSymbol   = func(lib ffi.Lib, name string, ret *ffi.Type, args ...*ffi.Type) (ffi.Fun, error) {
		return lib.Prep(name, ret, args...)
	}
)

// LibPath returns the directory from which Load opened the shared
// libraries. Empty until a successful Load.
func LibPath() string {
	loadMu.Lock()
	defer loadMu.Unlock()
	return libPath
}

// Load opens the LiteRT-LM shared libraries and binds C entry points.
//
//   - path is the directory containing the libs; "" triggers a search
//     of $LITERTLM_LIB and the platform default paths (see
//     loader.DefaultPaths).
//   - backend selects "cpu" or "gpu"; "" or unknown picks GPU then CPU.
//   - libName overrides the main C-API library short-name; "" falls
//     back to $LITERTLM_LIB_NAME, then the backend default.
//
// Auxiliary libs are dlopen'd before the main lib so DT_NEEDED references
// resolve without requiring LD_LIBRARY_PATH / DYLD_LIBRARY_PATH. Load is
// safe to call concurrently; subsequent successful calls are no-ops.
func Load(path, backend, libName string) error {
	loadMu.Lock()
	defer loadMu.Unlock()
	if loadedOnce {
		return nil
	}

	if libName == "" {
		libName = os.Getenv(envLibName)
	}

	if path == "" {
		searchName := libName
		if searchName == "" {
			searchName = defaultLibName(backend)
		}
		candidates := []string{os.Getenv(loader.EnvVar)}
		candidates = append(candidates, loader.DefaultPaths()...)
		found, err := loader.Find(searchName, candidates)
		if err != nil {
			return err
		}
		path = found
	}

	for _, name := range optionalLibs {
		auxLib, err := loadAuxLibrary(path, name)
		if err == nil && name == "LiteRtTopKWebGpuSampler" {
			if _, symErr := prepAuxSymbol(auxLib, "LiteRtTopKWebGpuSampler_UpdateConfig", &ffi.TypeVoid); symErr != nil {
				activeLevel := LogError
				if pendingLogLevel != nil {
					activeLevel = *pendingLogLevel
				}
				if activeLevel <= LogWarning {
					fmt.Fprintln(os.Stderr, "litertlm: warning: stale WebGPU sampler library detected (symbol LiteRtTopKWebGpuSampler_UpdateConfig is missing); WebGPU sampler is unavailable, falling back to CPU sampling")
				}
			}
		}
	}
	for _, name := range auxLibs {
		if _, err := loadAuxLibrary(path, name); err != nil {
			return err
		}
	}

	mainLib, err := loadMainLibVar(path, backend, libName)
	if err != nil {
		return err
	}
	if err := loadFuncsVar(mainLib); err != nil {
		return err
	}

	libPath = path
	loadedOnce = true
	applyPendingLogLevel()
	return nil
}

// defaultLibName returns the main C-API library short-name picked by
// the given backend. Used when libName is empty and a search needs a
// concrete filename to look for.
func defaultLibName(backend string) string {
	if short, ok := libByBackend[backend]; ok {
		return short
	}
	return "litertlm_c_cpu"
}

// loadMainLib opens the C-API library. When libName is non-empty it
// is used verbatim; otherwise the name is selected by backend, with a
// GPU-then-CPU fallback for unknown backends.
func loadMainLib(path, backend, libName string) (ffi.Lib, error) {
	if libName != "" {
		return loader.LoadLibrary(path, libName)
	}
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
