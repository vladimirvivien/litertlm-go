// Package litertlm is a purego-backed, cgo-free Go wrapper around
// Google's LiteRT-LM C API (see c/engine.h in the LiteRT-LM repository).
//
// The flow for local inference is:
//
//	litertlm.Load(libDir, "cpu", "")             // once; "" auto-picks
//	defer litertlm.Close()
//
//	s, _ := litertlm.NewEngineSettings(modelPath, "cpu", nil, nil)
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
	"cmp"
	"fmt"
	"os"
	"sync"

	"github.com/jupiterrider/ffi"

	"github.com/vladimirvivien/litertlm-go/pkg/loader"
)

// envLibName is consulted by Load when libName is empty.
const envLibName = "LITERTLM_LIB_NAME"

// defaultUnifiedLib is the unified C-API shared library produced by
// LiteRT-LM //c:litert-lm (liblitert-lm.so, liblitert-lm.dylib, litert-lm.dll).
const defaultUnifiedLib = "litert-lm"

// libByBackend maps the backend selector to legacy pre-1.0.0 C-API shared
// library short-names when falling back from defaultUnifiedLib.
var libByBackend = map[string]string{
	"cpu":     "litertlm_c_cpu",
	"gpu":     "litertlm_c",
	"apple":   "litertlm_c",
	"metal":   "litertlm_c",
	"ynnpack": "litertlm_c_cpu",
}

// optionalLibs are plugin / accelerator libraries that may accompany
// legacy builds. C API 1.0+ builds bundle dependencies inside liblitert-lm,
// so absence of these libraries is ignored.
var optionalLibs = []string{
	"GemmaModelConstraintProvider",
	"webgpu_dawn",
	"LiteRt",
	"LiteRtWebGpuAccelerator",
	"LiteRtTopKWebGpuSampler",
	"LiteRtMetalAccelerator", // macOS only
	"LiteRtTopKMetalSampler", // macOS only
}

// runtimeState holds process-wide native library, load, and API version state.
// loadMu serializes Load/SetMinLogLevel initialization; stateMu is a leaf lock
// held only for fast in-memory field reads/writes and never held across FFI.
type runtimeState struct {
	loadMu  sync.Mutex
	stateMu sync.RWMutex

	loadedOnce        bool
	path              string
	pendingLogLevel   *LogLevel
	libHandle         ffi.Lib
	realLibraryLoaded bool

	apiMajor         int
	apiMinor         int
	apiPatch         int
	apiVersionLoaded bool
}

var rt runtimeState

func (s *runtimeState) isLoaded() bool {
	s.stateMu.RLock()
	defer s.stateMu.RUnlock()
	return s.loadedOnce
}

func (s *runtimeState) isRealLibraryLoaded() bool {
	s.stateMu.RLock()
	defer s.stateMu.RUnlock()
	return s.realLibraryLoaded
}

func (s *runtimeState) libPath() string {
	s.stateMu.RLock()
	defer s.stateMu.RUnlock()
	return s.path
}

func (s *runtimeState) prepSymbol(name string, ret *ffi.Type, args ...*ffi.Type) (ffi.Fun, error) {
	s.stateMu.RLock()
	h := s.libHandle
	s.stateMu.RUnlock()
	return h.Prep(name, ret, args...)
}

func (s *runtimeState) setMainLib(lib ffi.Lib, real bool) {
	s.stateMu.Lock()
	s.libHandle = lib
	s.realLibraryLoaded = real
	s.stateMu.Unlock()
}

func (s *runtimeState) clearMainLib() {
	s.stateMu.Lock()
	s.libHandle = ffi.Lib{}
	s.realLibraryLoaded = false
	s.stateMu.Unlock()
}

func (s *runtimeState) markLoaded(path string) LogLevel {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	s.path = path
	s.loadedOnce = true
	level := LogError
	if s.pendingLogLevel != nil {
		level = *s.pendingLogLevel
		s.pendingLogLevel = nil
	}
	return level
}

func (s *runtimeState) apiVersion() (major, minor, patch int, ok bool) {
	s.stateMu.RLock()
	defer s.stateMu.RUnlock()
	return s.apiMajor, s.apiMinor, s.apiPatch, s.apiVersionLoaded
}

func (s *runtimeState) setAPIVersion(major, minor, patch int) {
	s.stateMu.Lock()
	s.apiMajor = major
	s.apiMinor = minor
	s.apiPatch = patch
	s.apiVersionLoaded = true
	s.stateMu.Unlock()
}

func (s *runtimeState) resetAPIVersion() {
	s.stateMu.Lock()
	s.apiMajor = 0
	s.apiMinor = 0
	s.apiPatch = 0
	s.apiVersionLoaded = false
	s.stateMu.Unlock()
}

var (
	loadAuxLibrary     = loader.LoadAuxLibrary
	loadMainLibVar     = loadMainLib
	loadFuncsVar       = loadFuncs
	checkAPIVersionVar = checkAPIVersion
	prepAuxSymbol      = func(lib ffi.Lib, name string, ret *ffi.Type, args ...*ffi.Type) (ffi.Fun, error) {
		return lib.Prep(name, ret, args...)
	}
)

// LibPath returns the directory from which Load opened the shared
// libraries. Empty until a successful Load.
func LibPath() string {
	return rt.libPath()
}

// Load opens the LiteRT-LM shared libraries and binds C entry points.
//
//   - path is the directory containing the libs; "" triggers a search
//     of $LITERTLM_LIB and the platform default paths (see
//     loader.DefaultPaths).
//   - backend selects "cpu" or "gpu"; "" or unknown selects the unified library.
//   - libName overrides the main C-API library short-name; "" falls
//     back to $LITERTLM_LIB_NAME, then "litert-lm".
//
// Load validates that the loaded shared library exports C API 1.x via
// litert_lm_get_api_version. Load is safe to call concurrently; subsequent
// successful calls are no-ops.
func Load(path, backend, libName string) error {
	rt.loadMu.Lock()
	defer rt.loadMu.Unlock()
	if rt.isLoaded() {
		return nil
	}

	if libName == "" {
		libName = os.Getenv(envLibName)
	}

	if path == "" {
		candidates := []string{os.Getenv(loader.EnvVar)}
		candidates = append(candidates, loader.DefaultPaths()...)
		searchName := cmp.Or(libName, defaultUnifiedLib)
		found, err := loader.Find(searchName, candidates)
		if err != nil && libName == "" {
			if legacyFound, legacyErr := loader.Find(legacyLibName(backend), candidates); legacyErr == nil {
				found = legacyFound
				err = nil
			}
		}
		if err != nil {
			return err
		}
		path = found
	}

	for _, name := range optionalLibs {
		auxLib, err := loadAuxLibrary(path, name)
		if err == nil && name == "LiteRtTopKWebGpuSampler" {
			if _, symErr := prepAuxSymbol(auxLib, "LiteRtTopKWebGpuSampler_UpdateConfig", &ffi.TypeVoid); symErr != nil {
				rt.stateMu.RLock()
				activeLevel := LogError
				if rt.pendingLogLevel != nil {
					activeLevel = *rt.pendingLogLevel
				}
				rt.stateMu.RUnlock()
				if activeLevel <= LogWarning {
					fmt.Fprintln(os.Stderr, "litertlm: warning: stale WebGPU sampler library detected (symbol LiteRtTopKWebGpuSampler_UpdateConfig is missing); WebGPU sampler is unavailable, falling back to CPU sampling")
				}
			}
		}
	}

	mainLib, err := loadMainLibVar(path, backend, libName)
	if err != nil {
		return err
	}
	if err := loadFuncsVar(mainLib); err != nil {
		return err
	}
	if rt.isRealLibraryLoaded() {
		if err := checkAPIVersionVar(); err != nil {
			rt.clearMainLib()
			return err
		}
	}

	level := rt.markLoaded(path)
	applyLogLevel(level)
	return nil
}

// legacyLibName returns the legacy pre-1.0.0 C-API library short-name
// selected by backend.
func legacyLibName(backend string) string {
	if short, ok := libByBackend[backend]; ok {
		return short
	}
	return "litertlm_c_cpu"
}

// loadMainLib opens the C-API library. When libName is non-empty it
// is used verbatim; otherwise the unified "litert-lm" library is loaded
// first, with legacy names as a last-resort fallback.
func loadMainLib(path, backend, libName string) (ffi.Lib, error) {
	if libName != "" {
		return loader.LoadLibrary(path, libName)
	}
	lib, err := loader.LoadLibrary(path, defaultUnifiedLib)
	if err == nil {
		return lib, nil
	}
	if short, ok := libByBackend[backend]; ok {
		if legacyLib, lerr := loader.LoadLibrary(path, short); lerr == nil {
			return legacyLib, nil
		}
	} else {
		if legacyLib, lerr := loader.LoadLibrary(path, "litertlm_c"); lerr == nil {
			return legacyLib, nil
		}
		if legacyLib, lerr := loader.LoadLibrary(path, "litertlm_c_cpu"); lerr == nil {
			return legacyLib, nil
		}
	}
	return ffi.Lib{}, err
}

// Close is a no-op retained for API symmetry: purego doesn't expose
// dlclose, and the C library's lifetime is tied to the process.
func Close() {}
