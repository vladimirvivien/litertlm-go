package litertlm

import "unsafe"

// SetMinLogLevel sets the minimum severity LiteRT-LM will emit.
// Use the LogInfo, LogWarning, LogError, LogFatal constants.
func SetMinLogLevel(level int) {
	v := int32(level)
	setMinLogLevelFunc.Call(nil, unsafe.Pointer(&v))
}
