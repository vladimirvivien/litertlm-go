// Package litertlm is a purego-backed, cgo-free Go wrapper around
// Google's LiteRT-LM C API (see c/engine.h in the LiteRT-LM repository).
//
// The flow for local inference is:
//
//	litertlm.Load(libDir)                        // once
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
// Ownership: every New*/Generate* result must be paired with .Delete(). No
// finalizers are attached. Strings returned by accessor methods are copied
// into Go memory, so they remain valid after their parent is deleted.
package litertlm

import (
	"github.com/vladimirvivien/litertlm-go/pkg/loader"
)

// mainLibShort is the "short name" (no prefix, no extension) of the C API
// shared library this package binds against. The loader resolves it to
// liblitertlm_c_cpu.so / .dylib / .dll per platform.
//
// If you built the GPU-capable variant (//c:litertlm_c), override this by
// copying the file alongside liblitertlm_c_cpu.{so,dylib}, or set
// LITERTLM_LIB_NAME before calling Load to select a different short name.
const mainLibShort = "litertlm_c_cpu"

// auxLibs are additional shared libraries the C API loads at runtime. They
// must be present in the same LITERTLM_LIB directory. libGemmaModelConstraintProvider
// is always required (even for CPU builds); the GPU-only plugins are listed
// but treated as optional — Load succeeds if they are absent.
var auxLibs = []string{
	"GemmaModelConstraintProvider", // always required
}

// optionalLibs are loaded if present but their absence is not an error. They
// are the GPU accelerator plugins shipped in LiteRT-LM's prebuilt/ directory.
var optionalLibs = []string{
	"LiteRt",
	"LiteRtWebGpuAccelerator",
	"LiteRtTopKWebGpuSampler",
	"LiteRtMetalAccelerator", // macOS only; missing on Linux, which is fine
}

var libPath string

// LibPath returns the directory from which the LiteRT-LM shared libraries
// were loaded. Empty until Load has been called successfully.
func LibPath() string { return libPath }

// Load dynamically opens the LiteRT-LM shared library set and binds every
// C entry point this package uses. `path` is the directory containing the
// shared libs; if empty, the LITERTLM_LIB environment variable is consulted.
//
// The required main library (liblitertlm_c_cpu.so/.dylib/.dll) and
// libGemmaModelConstraintProvider.* must be present. GPU accelerator plugins
// are loaded opportunistically — if they are not in the directory, Load still
// succeeds but backend="gpu" calls will fail at runtime.
//
// Auxiliary libraries are dlopen'd before the main library so that DT_NEEDED
// references in the main library resolve to the already-loaded copies. This
// lets Load work without the user having to set LD_LIBRARY_PATH /
// DYLD_LIBRARY_PATH.
func Load(path string) error {
	libPath = path

	// Optional libs first — skip silently if absent (CPU-only deployments).
	for _, name := range optionalLibs {
		_, _ = loader.LoadLibrary(path, name)
	}

	// Required aux libs before the main lib so ld.so finds them by soname.
	for _, name := range auxLibs {
		if _, err := loader.LoadLibrary(path, name); err != nil {
			return err
		}
	}

	mainLib, err := loader.LoadLibrary(path, mainLibShort)
	if err != nil {
		return err
	}

	return loadFuncs(mainLib)
}

// Close is a no-op retained for API symmetry with yzma. purego does not
// expose an explicit dlclose, and the C library's lifetime is tied to the
// process, so there is nothing to actively release here.
func Close() {}
