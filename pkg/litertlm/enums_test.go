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
