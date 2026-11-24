package litertlm

import (
	"errors"
	"math"
	"slices"
	"strings"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/jupiterrider/ffi"
)

func TestPhase2_ZeroHandleErrors(t *testing.T) {
	var es EngineSettings
	for name, err := range map[string]error{
		"SetSpeculativeDecodingConfig":  es.SetSpeculativeDecodingConfig(SpeculativeDecodingConfig{Mode: SpeculativeDecodingAdaptive}),
		"SetEnableYNNPACK":              es.SetEnableYNNPACK(true),
		"SetMaxVisionTokensPerImage":    es.SetMaxVisionTokensPerImage(256),
		"SetGPUEnableMetalResidencySet": es.SetGPUEnableMetalResidencySet(true),
		"SetGPUDecodeStepsPerSync":      es.SetGPUDecodeStepsPerSync(4),
		"SetGPUWaitForWeightUploads":    es.SetGPUWaitForWeightUploads(true),
		"SetUseRingbuffersLocalAtt":     es.SetUseRingbuffersLocalAttention(true),
		"SetSingleThreadedExecution":    es.SetSingleThreadedExecution(true),
		"SetSelectedSignatures":         es.SetSelectedSignatures([]string{"prefill_1024", "decode"}),
	} {
		if err == nil {
			t.Errorf("EngineSettings(0).%s = nil, want error", name)
		}
	}

	var sc SessionConfig
	if err := sc.SetSpeculativeDecodingConfig(SpeculativeDecodingConfig{Mode: SpeculativeDecodingOn}); err == nil {
		t.Error("SessionConfig(0).SetSpeculativeDecodingConfig = nil, want error")
	}

	var s Session
	if _, err := s.Clone(); err == nil {
		t.Error("Session(0).Clone = nil, want error")
	}
	for name, err := range map[string]error{
		"SaveCheckpoint":     s.SaveCheckpoint("cp"),
		"RewindToCheckpoint": s.RewindToCheckpoint("cp"),
		"RewindToStep":       s.RewindToStep(1),
		"RunDecodeAsync":     s.RunDecodeAsync(func(StreamChunk) {}),
	} {
		if err == nil {
			t.Errorf("Session(0).%s = nil, want error", name)
		}
	}

	var validSess Session = 1
	if err := validSess.RunDecodeAsync(nil); err == nil || !strings.Contains(err.Error(), "nil callback") {
		t.Errorf("Session.RunDecodeAsync(nil) = %v, want nil callback error", err)
	}

	var cc ConversationConfig
	for name, err := range map[string]error{
		"SetPromptTemplate":     cc.SetPromptTemplate("{{.}}"),
		"SetConstraintProvider": cc.SetConstraintProvider(ConstraintProviderLLGuidance),
		"SetThinkingConfig":     cc.SetThinkingConfig(ThinkingConfig{Enabled: true, TokenBudget: 64}),
	} {
		if err == nil {
			t.Errorf("ConversationConfig(0).%s = nil, want error", name)
		}
	}

	var conv Conversation
	if err := conv.WaitUntilDone(); err == nil {
		t.Error("Conversation(0).WaitUntilDone = nil, want error")
	}

	var oa OptionalArgs
	for name, err := range map[string]error{
		"SetRepetitionPenalty": oa.SetRepetitionPenalty(nil),
		"SetNoRepeatNgram":     oa.SetNoRepeatNgram(nil),
		"SetSuppressTokens":    oa.SetSuppressTokens(nil),
		"SetThinkingConfig":    oa.SetThinkingConfig(nil),
		"SetConstraint":        oa.SetConstraint(ConstraintRegex, "yes|no"),
		"SetToolChoice":        oa.SetToolChoice(`"auto"`),
	} {
		if err == nil {
			t.Errorf("OptionalArgs(0).%s = nil, want error", name)
		}
	}

	var bi BenchmarkInfo
	if got := bi.MaxNumTokens(); got != 0 {
		t.Errorf("BenchmarkInfo(0).MaxNumTokens() = %d, want 0", got)
	}
}

func TestPhase2_APIVersion11Gates(t *testing.T) {
	t.Cleanup(resetAPIVersion)
	rt.setAPIVersion(1, 0, 0)

	es := EngineSettings(1)
	sigErr := es.SetSelectedSignatures([]string{"prefill_1024", "decode"})
	if !errors.Is(sigErr, ErrUnimplemented) {
		t.Errorf("SetSelectedSignatures on C API 1.0.0 = %v, want ErrUnimplemented", sigErr)
	}

	oa := OptionalArgs(1)
	tcErr := oa.SetToolChoice(`"required"`)
	if !errors.Is(tcErr, ErrUnimplemented) {
		t.Errorf("SetToolChoice on C API 1.0.0 = %v, want ErrUnimplemented", tcErr)
	}
}

func TestPhase2_OptionsAndCloning(t *testing.T) {
	sigs := []string{"prefill_1024", "decode"}
	var cfg clientConfig
	for _, opt := range []Option{
		WithSpeculativeDecoding(SpeculativeDecodingConfig{Mode: SpeculativeDecodingAdaptive, Lenience: 0.3}),
		WithYNNPACK(true),
		WithMaxVisionTokensPerImage(256),
		WithMetalResidencySet(true),
		WithGPUDecodeStepsPerSync(2),
		WithGPUWaitForWeightUploads(false),
		WithRingbuffersLocalAttention(true),
		WithSingleThreadedExecution(true),
		WithSelectedSignatures(sigs...),
	} {
		opt(&cfg)
	}
	sigs[0] = "mutated"

	if cfg.speculativeDecoding == nil || cfg.speculativeDecoding.Mode != SpeculativeDecodingAdaptive || cfg.speculativeDecoding.Lenience != 0.3 {
		t.Errorf("speculativeDecoding = %+v", cfg.speculativeDecoding)
	}
	if cfg.enableYNNPACK == nil || !*cfg.enableYNNPACK {
		t.Errorf("enableYNNPACK = %v", cfg.enableYNNPACK)
	}
	if cfg.maxVisionTokensPerImage == nil || *cfg.maxVisionTokensPerImage != 256 {
		t.Errorf("maxVisionTokensPerImage = %v", cfg.maxVisionTokensPerImage)
	}
	if cfg.metalResidencySet == nil || !*cfg.metalResidencySet {
		t.Errorf("metalResidencySet = %v", cfg.metalResidencySet)
	}
	if cfg.gpuDecodeStepsPerSync == nil || *cfg.gpuDecodeStepsPerSync != 2 {
		t.Errorf("gpuDecodeStepsPerSync = %v", cfg.gpuDecodeStepsPerSync)
	}
	if cfg.gpuWaitForWeightUploads == nil || *cfg.gpuWaitForWeightUploads {
		t.Errorf("gpuWaitForWeightUploads = %v", cfg.gpuWaitForWeightUploads)
	}
	if cfg.useRingbuffersLocalAttn == nil || !*cfg.useRingbuffersLocalAttn {
		t.Errorf("useRingbuffersLocalAttn = %v", cfg.useRingbuffersLocalAttn)
	}
	if cfg.singleThreadedExecution == nil || !*cfg.singleThreadedExecution {
		t.Errorf("singleThreadedExecution = %v", cfg.singleThreadedExecution)
	}
	if !slices.Equal(cfg.selectedSignatures, []string{"prefill_1024", "decode"}) {
		t.Errorf("selectedSignatures = %v, want defensive clone", cfg.selectedSignatures)
	}

	var chCfg chatConfig
	WithPromptTemplate("custom {{.}}")(&chCfg)
	WithConstraintProvider(ConstraintProviderLLGuidance)(&chCfg)
	WithThinking(ThinkingConfig{Enabled: true, TokenBudget: 128})(&chCfg)
	if chCfg.promptTemplate != "custom {{.}}" {
		t.Errorf("promptTemplate = %q", chCfg.promptTemplate)
	}
	if chCfg.constraintProvider == nil || *chCfg.constraintProvider != ConstraintProviderLLGuidance {
		t.Errorf("constraintProvider = %v", chCfg.constraintProvider)
	}
	if chCfg.thinking == nil || !chCfg.thinking.Enabled || chCfg.thinking.TokenBudget != 128 {
		t.Errorf("thinking = %+v", chCfg.thinking)
	}

	ids := []int{10, 20}
	rtCfg := resolveRuntimeConfig([]RuntimeOption{
		WithTurnThinking(ThinkingConfig{Enabled: false}),
		WithConstraint(ConstraintRegex, `[0-9]+`),
		WithToolChoice(ToolChoiceFunction("lookup")),
		WithRepetitionPenalty(RepetitionPenaltyConfig{RepetitionPenalty: 1.2, PresencePenalty: 0.1, FrequencyPenalty: 0.05, WindowSize: 32}),
		WithNoRepeatNgram(NoRepeatNgramConfig{NgramSize: 3, WindowSize: 64}),
		WithSuppressTokens(ids...),
	})
	ids[0] = 999

	if rtCfg.thinking == nil || rtCfg.thinking.Enabled {
		t.Errorf("turn thinking = %+v", rtCfg.thinking)
	}
	if rtCfg.constraintType != ConstraintRegex || rtCfg.constraintPattern != `[0-9]+` {
		t.Errorf("constraint = (%v, %q)", rtCfg.constraintType, rtCfg.constraintPattern)
	}
	if rtCfg.toolChoice.IsZero() {
		t.Error("expected non-zero toolChoice")
	}
	if rtCfg.repetitionPenalty == nil || rtCfg.repetitionPenalty.RepetitionPenalty != 1.2 {
		t.Errorf("repetitionPenalty = %+v", rtCfg.repetitionPenalty)
	}
	if rtCfg.noRepeatNgram == nil || rtCfg.noRepeatNgram.NgramSize != 3 {
		t.Errorf("noRepeatNgram = %+v", rtCfg.noRepeatNgram)
	}
	if rtCfg.suppressTokens == nil || !slices.Equal(rtCfg.suppressTokens.TokenIDs, []int{10, 20}) {
		t.Errorf("suppressTokens = %+v, want defensive clone", rtCfg.suppressTokens)
	}
}

func TestToolChoice_JSON(t *testing.T) {
	tests := []struct {
		name    string
		tc      ToolChoice
		want    string
		wantErr bool
	}{
		{"zero", ToolChoice{}, "", false},
		{"auto", ToolChoiceAuto(), `"auto"`, false},
		{"none", ToolChoiceNone(), `"none"`, false},
		{"required", ToolChoiceRequired(), `"required"`, false},
		{"function", ToolChoiceFunction("get_weather"), `{"function":{"name":"get_weather"},"type":"function"}`, false},
		{"allowed_tools", ToolChoiceAllowedTools("required", "a", "b"), `{"allowed_tools":{"mode":"required","tools":[{"function":{"name":"a"},"type":"function"},{"function":{"name":"b"},"type":"function"}]},"type":"allowed_tools"}`, false},
		{"allowed_tools_default_mode", ToolChoiceAllowedTools("", "only_tool"), `{"allowed_tools":{"mode":"auto","tools":[{"function":{"name":"only_tool"},"type":"function"}]},"type":"allowed_tools"}`, false},
		{"allowed_tools_empty_names", ToolChoiceAllowedTools("auto"), "", true},
		{"raw_json", ToolChoiceRawJSON(`"none"`), `"none"`, false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.tc.JSON()
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got %q", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("JSON(): %v", err)
			}
			if got != tc.want {
				t.Errorf("JSON() = %s, want %s", got, tc.want)
			}
		})
	}
}

func TestPhase2_GenericFFIHelpers(t *testing.T) {
	t.Run("setHandleConfig", func(t *testing.T) {
		var gotTarget, gotCfg uintptr
		cb := purego.NewCallback(func(h, cfg uintptr) uintptr {
			gotTarget = h
			gotCfg = cfg
			return uintptr(StatusOk)
		})
		var cif ffi.Cif
		if st := ffi.PrepCif(&cif, ffi.DefaultAbi, 2, &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer); st != ffi.OK {
			t.Fatalf("PrepCif: %v", st)
		}
		lf := &lazyFun{
			name: "stub_set_config",
			ret:  &ffi.TypeSint32,
			args: []*ffi.Type{&ffi.TypePointer, &ffi.TypePointer},
			fun:  ffi.Fun{Addr: cb, Cif: &cif},
		}
		lf.once.Do(func() {})

		if err := setHandleConfig(EngineSettings(0xAA), "settings", lf, "stub_set_config", thinkingConfigHandle(0xBB)); err != nil {
			t.Fatalf("setHandleConfig(non-nil): %v", err)
		}
		if gotTarget != 0xAA || gotCfg != 0xBB {
			t.Errorf("got (%#x, %#x), want (0xaa, 0xbb)", gotTarget, gotCfg)
		}
		if err := setHandleConfig(EngineSettings(0xAA), "settings", lf, "stub_set_config", thinkingConfigHandle(0)); err != nil {
			t.Fatalf("setHandleConfig(0): %v", err)
		}
		if gotCfg != 0 {
			t.Errorf("expected nil config pointer 0, got %#x", gotCfg)
		}
	})

	t.Run("setHandleStringSlice", func(t *testing.T) {
		var gotN uint64
		var firstPtr *byte
		cb := purego.NewCallback(func(h uintptr, arr **byte, n uint64) uintptr {
			gotN = n
			if n > 0 && arr != nil {
				s := unsafe.Slice(arr, n)
				firstPtr = s[0]
			}
			return uintptr(StatusOk)
		})
		var cif ffi.Cif
		if st := ffi.PrepCif(&cif, ffi.DefaultAbi, 3, &ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffiTypeSizeT); st != ffi.OK {
			t.Fatalf("PrepCif: %v", st)
		}
		lf := &lazyFun{
			name: "stub_set_strs",
			ret:  &ffi.TypeSint32,
			args: []*ffi.Type{&ffi.TypePointer, &ffi.TypePointer, &ffiTypeSizeT},
			fun:  ffi.Fun{Addr: cb, Cif: &cif},
		}
		lf.once.Do(func() {})

		if err := setHandleStringSlice(EngineSettings(1), "settings", lf, "stub_set_strs", nil); err == nil {
			t.Fatal("expected error for empty string slice")
		}
		if err := setHandleStringSlice(EngineSettings(1), "settings", lf, "stub_set_strs", []string{"prefill_1024", "decode"}); err != nil {
			t.Fatalf("setHandleStringSlice: %v", err)
		}
		if gotN != 2 || firstPtr == nil {
			t.Errorf("got (n=%d, firstPtr=%v), want (2, non-nil)", gotN, firstPtr)
		}
	})

	t.Run("withTempHandle_cleans_up", func(t *testing.T) {
		deleted := false
		err := withTempHandle(
			func() (thinkingConfigHandle, error) { return thinkingConfigHandle(0x42), nil },
			func(h thinkingConfigHandle) { deleted = h == 0x42 },
			func(h thinkingConfigHandle) error { return nil },
		)
		if err != nil || !deleted {
			t.Errorf("withTempHandle: err=%v deleted=%v", err, deleted)
		}
	})

	t.Run("needsLenienceCall", func(t *testing.T) {
		if needsLenienceCall(0) {
			t.Error("needsLenienceCall(0) = true, want false")
		}
		if !needsLenienceCall(0.25) {
			t.Error("needsLenienceCall(0.25) = false, want true")
		}
		if !needsLenienceCall(-0.1) {
			t.Error("needsLenienceCall(-0.1) = false, want true")
		}
		if !needsLenienceCall(float32(math.NaN())) {
			t.Error("needsLenienceCall(NaN) = false, want true")
		}
	})
}
