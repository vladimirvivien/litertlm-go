package litertlm

import (
	"fmt"
	"os"
	"unsafe"
)

// SetMinLogLevel sets the process-global minimum severity LiteRT-LM
// will emit. Pass any of LogVerbose, LogDebug, LogInfo, LogWarning,
// LogError, LogFatal, or LogQuiet. The C side defaults to LogInfo if
// SetMinLogLevel is never called; examples in this module call
// SetMinLogLevel(LogQuiet) at the top of main to silence the loader
// and executor init chatter.
//
// May be called before the LiteRT-LM library has been loaded. The
// value is buffered and applied when Load (or the Load that runs
// inside New) returns successfully, so the level takes effect before
// the first chatty C call (engine settings, engine construction).
//
// Never panics. If the loaded library does not export
// litert_lm_set_min_log_level (an older build), the level request is
// dropped with a single warning to stderr and the engine retains its
// previous threshold.
func SetMinLogLevel(level LogLevel) {
	loadMu.Lock()
	defer loadMu.Unlock()
	if !loadedOnce {
		pendingLogLevel = &level
		return
	}
	applyLogLevelLocked(level)
}

// applyPendingLogLevel is called from Load after the C entry points
// are bound. loadMu must be held by the caller.
func applyPendingLogLevel() {
	if pendingLogLevel == nil {
		return
	}
	level := *pendingLogLevel
	pendingLogLevel = nil
	applyLogLevelLocked(level)
}

// applyLogLevelLocked invokes the C-side setter and absorbs any
// panic from a missing symbol so SetMinLogLevel never propagates one
// to the caller. loadMu must be held.
func applyLogLevelLocked(level LogLevel) {
	defer func() {
		if r := recover(); r != nil {
			fmt.Fprintf(os.Stderr, "litertlm: SetMinLogLevel(%s) dropped: %v\n", level, r)
		}
	}()
	setMinLogLevelFunc.Call(nil, unsafe.Pointer(new(int32(level))))
}
