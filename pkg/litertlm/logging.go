package litertlm

import "unsafe"

// SetMinLogLevel sets the minimum severity LiteRT-LM will emit.
// Use the LogVerbose, LogDebug, LogInfo, LogWarning, LogError, LogFatal, or
// LogSilent constants (see c/engine.h for the underlying mapping).
func SetMinLogLevel(level int) {
	setMinLogLevelFunc.Call(nil, unsafe.Pointer(new(int32(level))))
}
