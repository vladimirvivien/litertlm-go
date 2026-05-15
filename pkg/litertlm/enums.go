package litertlm

import "strconv"

// InputDataType identifies the kind of data in an InputData record.
// Values mirror the `InputDataType` enum in c/engine.h.
type InputDataType int32

const (
	InputText     InputDataType = 0
	InputImage    InputDataType = 1
	InputImageEnd InputDataType = 2
	InputAudio    InputDataType = 3
	InputAudioEnd InputDataType = 4
)

// SamplerType mirrors the `Type` enum for LiteRtLmSamplerParams in c/engine.h.
type SamplerType int32

const (
	SamplerTypeUnspecified SamplerType = 0
	SamplerTopK            SamplerType = 1
	SamplerTopP            SamplerType = 2
	SamplerGreedy          SamplerType = 3
)

// LogLevel is the severity floor used by SetMinLogLevel, mirroring
// the levels documented in c/engine.h next to litert_lm_set_min_log_level.
type LogLevel int32

const (
	LogVerbose LogLevel = 0
	LogDebug   LogLevel = 1
	LogInfo    LogLevel = 2
	LogWarning LogLevel = 3
	LogError   LogLevel = 4
	LogFatal   LogLevel = 5
	LogQuiet   LogLevel = 1000
)

// String returns the human-readable name of the level. Unknown values
// render as "LogLevel(<n>)".
func (l LogLevel) String() string {
	switch l {
	case LogVerbose:
		return "Verbose"
	case LogDebug:
		return "Debug"
	case LogInfo:
		return "Info"
	case LogWarning:
		return "Warning"
	case LogError:
		return "Error"
	case LogFatal:
		return "Fatal"
	case LogQuiet:
		return "Quiet"
	default:
		return "LogLevel(" + strconv.Itoa(int(l)) + ")"
	}
}
