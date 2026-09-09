package litertlm

import (
	"fmt"
	"os"
	"unsafe"
)

// SetMinLogLevel sets the process-global minimum severity LiteRT-LM
// will emit. Pass any of LogVerbose, LogDebug, LogInfo, LogWarning,
// LogError, LogFatal, or LogQuiet. On successful Load, the wrapper sets the
// default floor to LogError unless SetMinLogLevel was called beforehand;
// examples in this module typically call SetMinLogLevel(LogQuiet) before
// Load / New to silence loader and executor chatter.
//
// May be called before the LiteRT-LM library has been loaded. The value is
// buffered and applied when Load completes.
//
// If the loaded library does not export litert_lm_set_min_log_level, the
// request is dropped with a single warning to stderr and the engine retains
// its previous threshold.
func SetMinLogLevel(level LogLevel) {
	rt.loadMu.Lock()
	defer rt.loadMu.Unlock()
	if !rt.isLoaded() {
		rt.stateMu.Lock()
		rt.pendingLogLevel = new(level)
		rt.stateMu.Unlock()
		return
	}
	applyLogLevel(level)
}

// applyLogLevel invokes the C-side setter and logs to stderr if the symbol
// is unavailable in the loaded library.
func applyLogLevel(level LogLevel) {
	var status int32
	if err := setMinLogLevelFunc.callErr(unsafe.Pointer(&status), unsafe.Pointer(new(int32(level)))); err != nil {
		fmt.Fprintf(os.Stderr, "litertlm: SetMinLogLevel(%s) dropped: %v\n", level, err)
	}
}
