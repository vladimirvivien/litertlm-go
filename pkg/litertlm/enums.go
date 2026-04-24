package litertlm

// InputDataType identifies the kind of data in an InputData record.
// Values mirror the `InputDataType` enum in c/engine.h.
type InputDataType int32

const (
	InputText       InputDataType = 0
	InputImage      InputDataType = 1
	InputImageEnd   InputDataType = 2
	InputAudio      InputDataType = 3
	InputAudioEnd   InputDataType = 4
)

// SamplerType mirrors the `Type` enum for LiteRtLmSamplerParams in c/engine.h.
type SamplerType int32

const (
	SamplerTypeUnspecified SamplerType = 0
	SamplerTopK            SamplerType = 1
	SamplerTopP            SamplerType = 2
	SamplerGreedy          SamplerType = 3
)

// Log severity levels accepted by SetMinLogLevel, mirroring the levels
// documented in c/engine.h next to litert_lm_set_min_log_level.
const (
	LogVerbose = 0
	LogDebug   = 1
	LogInfo    = 2
	LogWarning = 3
	LogError   = 4
	LogFatal   = 5
	LogSilent  = 1000
)
