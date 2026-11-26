package litertlm

import "testing"

// These tests pin the enum values to the documented C constants. Drift
// would silently corrupt FFI calls — a wrong InputText would be parsed
// as a different content type by the C side.

func TestInputDataTypeValues(t *testing.T) {
	tests := []struct {
		name string
		got  InputDataType
		want int32
	}{
		{"InputText", InputText, 0},
		{"InputImage", InputImage, 1},
		{"InputImageEnd", InputImageEnd, 2},
		{"InputAudio", InputAudio, 3},
		{"InputAudioEnd", InputAudioEnd, 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if int32(tt.got) != tt.want {
				t.Errorf("%s = %d, want %d", tt.name, tt.got, tt.want)
			}
		})
	}
}

func TestSamplerTypeValues(t *testing.T) {
	tests := []struct {
		name string
		got  SamplerType
		want int32
	}{
		{"SamplerTypeUnspecified", SamplerTypeUnspecified, 0},
		{"SamplerTopK", SamplerTopK, 1},
		{"SamplerTopP", SamplerTopP, 2},
		{"SamplerGreedy", SamplerGreedy, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if int32(tt.got) != tt.want {
				t.Errorf("%s = %d, want %d", tt.name, tt.got, tt.want)
			}
		})
	}
}

func TestLogLevels(t *testing.T) {
	tests := []struct {
		name string
		got  LogLevel
		want int32
	}{
		{"LogVerbose", LogVerbose, 0},
		{"LogDebug", LogDebug, 1},
		{"LogInfo", LogInfo, 2},
		{"LogWarning", LogWarning, 3},
		{"LogError", LogError, 4},
		{"LogFatal", LogFatal, 5},
		{"LogQuiet", LogQuiet, 1000},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if int32(tt.got) != tt.want {
				t.Errorf("%s = %d, want %d", tt.name, tt.got, tt.want)
			}
		})
	}
}

func TestLogLevel_String(t *testing.T) {
	cases := []struct {
		got  LogLevel
		want string
	}{
		{LogVerbose, "Verbose"},
		{LogDebug, "Debug"},
		{LogInfo, "Info"},
		{LogWarning, "Warning"},
		{LogError, "Error"},
		{LogFatal, "Fatal"},
		{LogQuiet, "Quiet"},
		{LogLevel(42), "LogLevel(42)"},
	}
	for _, c := range cases {
		if got := c.got.String(); got != c.want {
			t.Errorf("LogLevel(%d).String() = %q, want %q", c.got, got, c.want)
		}
	}
}

func TestModalityValues(t *testing.T) {
	tests := []struct {
		name string
		got  Modality
		want int32
	}{
		{"ModalityText", ModalityText, 0},
		{"ModalityVision", ModalityVision, 1},
		{"ModalityAudio", ModalityAudio, 2},
		{"ModalityVideo", ModalityVideo, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if int32(tt.got) != tt.want {
				t.Errorf("%s = %d, want %d", tt.name, tt.got, tt.want)
			}
		})
	}
}

func TestModality_String(t *testing.T) {
	cases := []struct {
		got  Modality
		want string
	}{
		{ModalityText, "Text"},
		{ModalityVision, "Vision"},
		{ModalityAudio, "Audio"},
		{ModalityVideo, "Video"},
		{Modality(99), "Modality(99)"},
	}
	for _, c := range cases {
		if got := c.got.String(); got != c.want {
			t.Errorf("Modality(%d).String() = %q, want %q", c.got, got, c.want)
		}
	}
}

func TestPhase2EnumValuesAndStrings(t *testing.T) {
	specCases := []struct {
		got  SpeculativeDecodingMode
		raw  int32
		want string
	}{
		{SpeculativeDecodingOff, 0, "Off"},
		{SpeculativeDecodingOn, 1, "On"},
		{SpeculativeDecodingAdaptive, 2, "Adaptive"},
		{SpeculativeDecodingMode(9), 9, "SpeculativeDecodingMode(9)"},
	}
	for _, tc := range specCases {
		if int32(tc.got) != tc.raw || tc.got.String() != tc.want {
			t.Errorf("SpeculativeDecodingMode(%d) = %q, want %q", tc.got, tc.got.String(), tc.want)
		}
	}

	constraintCases := []struct {
		got  ConstraintType
		raw  int32
		want string
	}{
		{ConstraintNone, 0, "None"},
		{ConstraintRegex, 1, "Regex"},
		{ConstraintJSONSchema, 2, "JsonSchema"},
		{ConstraintType(9), 9, "ConstraintType(9)"},
	}
	for _, tc := range constraintCases {
		if int32(tc.got) != tc.raw || tc.got.String() != tc.want {
			t.Errorf("ConstraintType(%d) = %q, want %q", tc.got, tc.got.String(), tc.want)
		}
	}

	providerCases := []struct {
		got  ConstraintProviderType
		raw  int32
		want string
	}{
		{ConstraintProviderNone, 0, "None"},
		{ConstraintProviderLLGuidance, 1, "LlGuidance"},
		{ConstraintProviderType(9), 9, "ConstraintProviderType(9)"},
	}
	for _, tc := range providerCases {
		if int32(tc.got) != tc.raw || tc.got.String() != tc.want {
			t.Errorf("ConstraintProviderType(%d) = %q, want %q", tc.got, tc.got.String(), tc.want)
		}
	}
}

func TestPhase3aEnumValuesAndStrings(t *testing.T) {
	modelCases := []struct {
		got  ModelType
		raw  int32
		want string
	}{
		{ModelTypeUnknown, 0, "Unknown"},
		{ModelTypeLLM, 1, "LLM"},
		{ModelTypeEmbedding, 2, "Embedding"},
		{ModelType(9), 9, "ModelType(9)"},
	}
	for _, tc := range modelCases {
		if int32(tc.got) != tc.raw || tc.got.String() != tc.want {
			t.Errorf("ModelType(%d) = %q, want %q", tc.got, tc.got.String(), tc.want)
		}
	}

	backendCases := []struct {
		got  BackendType
		raw  int32
		want string
	}{
		{BackendTypeCPU, 1, "CPU"},
		{BackendTypeGPU, 2, "GPU"},
		{BackendTypeNPU, 3, "NPU"},
		{BackendType(9), 9, "BackendType(9)"},
	}
	for _, tc := range backendCases {
		if int32(tc.got) != tc.raw || tc.got.String() != tc.want {
			t.Errorf("BackendType(%d) = %q, want %q", tc.got, tc.got.String(), tc.want)
		}
	}

	npuCases := []struct {
		got  NPUBrand
		raw  int32
		want string
	}{
		{NPUBrandUnknown, 0, "Unknown"},
		{NPUBrandQualcomm, 1, "Qualcomm"},
		{NPUBrandGoogleTensor, 2, "GoogleTensor"},
		{NPUBrandMediaTek, 3, "MediaTek"},
		{NPUBrandIntel, 4, "Intel"},
		{NPUBrandSamsung, 5, "Samsung"},
		{NPUBrand(9), 9, "NPUBrand(9)"},
	}
	for _, tc := range npuCases {
		if int32(tc.got) != tc.raw || tc.got.String() != tc.want {
			t.Errorf("NPUBrand(%d) = %q, want %q", tc.got, tc.got.String(), tc.want)
		}
	}

	overflowCases := []struct {
		got  OverflowStrategy
		raw  int32
		want string
	}{
		{OverflowChunkAndAverage, 0, "ChunkAndAverage"},
		{OverflowTruncate, 1, "Truncate"},
		{OverflowError, 2, "Error"},
		{OverflowStrategy(9), 9, "OverflowStrategy(9)"},
	}
	for _, tc := range overflowCases {
		if int32(tc.got) != tc.raw || tc.got.String() != tc.want {
			t.Errorf("OverflowStrategy(%d) = %q, want %q", tc.got, tc.got.String(), tc.want)
		}
	}
}
