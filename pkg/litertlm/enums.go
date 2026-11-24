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

// Modality identifies input and output modalities supported by LiteRT-LM models.
// Values mirror the `LiteRtLmModality` enum in c/capabilities.h.
type Modality int32

const (
	ModalityText   Modality = 0
	ModalityVision Modality = 1
	ModalityAudio  Modality = 2
	ModalityVideo  Modality = 3
)

// String returns the human-readable name of the modality. Unknown values
// render as "Modality(<n>)".
func (m Modality) String() string {
	switch m {
	case ModalityText:
		return "Text"
	case ModalityVision:
		return "Vision"
	case ModalityAudio:
		return "Audio"
	case ModalityVideo:
		return "Video"
	default:
		return "Modality(" + strconv.Itoa(int(m)) + ")"
	}
}

// SpeculativeDecodingMode mirrors `LiteRtLmSpeculativeDecodingMode` in c/engine.h.
type SpeculativeDecodingMode int32

const (
	SpeculativeDecodingOff      SpeculativeDecodingMode = 0
	SpeculativeDecodingOn       SpeculativeDecodingMode = 1
	SpeculativeDecodingAdaptive SpeculativeDecodingMode = 2
)

// String returns the human-readable name of the speculative decoding mode.
func (m SpeculativeDecodingMode) String() string {
	switch m {
	case SpeculativeDecodingOff:
		return "Off"
	case SpeculativeDecodingOn:
		return "On"
	case SpeculativeDecodingAdaptive:
		return "Adaptive"
	default:
		return "SpeculativeDecodingMode(" + strconv.Itoa(int(m)) + ")"
	}
}

// ConstraintType mirrors `LiteRtLmConstraintType` in c/conversation.h.
type ConstraintType int32

const (
	ConstraintNone       ConstraintType = 0
	ConstraintRegex      ConstraintType = 1
	ConstraintJSONSchema ConstraintType = 2
)

// String returns the human-readable name of the constraint type.
func (c ConstraintType) String() string {
	switch c {
	case ConstraintNone:
		return "None"
	case ConstraintRegex:
		return "Regex"
	case ConstraintJSONSchema:
		return "JsonSchema"
	default:
		return "ConstraintType(" + strconv.Itoa(int(c)) + ")"
	}
}

// ConstraintProviderType mirrors `LiteRtLmConstraintProviderType` in c/conversation.h.
type ConstraintProviderType int32

const (
	ConstraintProviderNone       ConstraintProviderType = 0
	ConstraintProviderLLGuidance ConstraintProviderType = 1
)

// String returns the human-readable name of the constraint provider type.
func (c ConstraintProviderType) String() string {
	switch c {
	case ConstraintProviderNone:
		return "None"
	case ConstraintProviderLLGuidance:
		return "LlGuidance"
	default:
		return "ConstraintProviderType(" + strconv.Itoa(int(c)) + ")"
	}
}
