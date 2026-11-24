package litertlm_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func TestLive_Phase2_EngineSettingsAndBenchmark(t *testing.T) {
	libDir, modelPath := requireTestModel(t)
	if err := litertlm.Load(libDir, "cpu", ""); err != nil {
		t.Fatalf("Load: %v", err)
	}
	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	settings, err := litertlm.NewEngineSettings(modelPath, "cpu", nil, nil)
	if err != nil {
		t.Fatalf("NewEngineSettings: %v", err)
	}
	defer settings.Delete()

	for name, setErr := range map[string]error{
		"SetEnableYNNPACK":              settings.SetEnableYNNPACK(false),
		"SetMaxVisionTokensPerImage":    settings.SetMaxVisionTokensPerImage(256),
		"SetGPUEnableMetalResidencySet": settings.SetGPUEnableMetalResidencySet(false),
		"SetGPUDecodeStepsPerSync":      settings.SetGPUDecodeStepsPerSync(1),
		"SetGPUWaitForWeightUploads":    settings.SetGPUWaitForWeightUploads(false),
		"SetUseRingbuffersLocalAtt":     settings.SetUseRingbuffersLocalAttention(false),
		"SetSingleThreadedExecution":    settings.SetSingleThreadedExecution(false),
		"SetSpeculativeDecodingConfig":  settings.SetSpeculativeDecodingConfig(litertlm.SpeculativeDecodingConfig{Mode: litertlm.SpeculativeDecodingAdaptive, Lenience: 0}),
	} {
		if setErr != nil {
			t.Errorf("EngineSettings.%s: %v", name, setErr)
		}
	}

	maj, minVer, _, _ := litertlm.APIVersion()
	sigErr := settings.SetSelectedSignatures([]string{"prefill_1024", "decode"})
	if maj == 1 && minVer < 1 && !errors.Is(sigErr, litertlm.ErrUnimplemented) {
		t.Errorf("SetSelectedSignatures on %d.%d = %v, want ErrUnimplemented", maj, minVer, sigErr)
	}

	lenErr := settings.SetSpeculativeDecodingConfig(litertlm.SpeculativeDecodingConfig{
		Mode:     litertlm.SpeculativeDecodingAdaptive,
		Lenience: 0.25,
	})
	if maj == 1 && minVer < 1 {
		if !errors.Is(lenErr, litertlm.ErrUnimplemented) {
			t.Errorf("SetSpeculativeDecodingConfig(lenience=0.25) on %d.%d = %v, want ErrUnimplemented", maj, minVer, lenErr)
		}
	} else if lenErr != nil {
		t.Errorf("SetSpeculativeDecodingConfig(lenience=0.25): %v", lenErr)
	}

	client, err := litertlm.New(context.Background(),
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
		litertlm.WithBackend("cpu"),
		litertlm.WithMaxTokens(1024),
		litertlm.WithBenchmarkEnabled(),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = client.Close() }()

	resp, err := client.GenerateResponse(context.Background(), "Name one primary color.", litertlm.WithMaxOutputTokens(4))
	if err != nil {
		t.Fatalf("GenerateResponse: %v", err)
	}
	b := resp.Benchmark()
	if b == nil {
		t.Fatal("expected non-nil Benchmark")
	}
	if b.MaxNumTokens <= 0 {
		t.Errorf("Benchmark.MaxNumTokens = %d, want > 0", b.MaxNumTokens)
	}
}

func TestLive_Phase2_SessionCheckpointsCloneAsync(t *testing.T) {
	libDir, modelPath := requireTestModel(t)
	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	client, err := litertlm.New(context.Background(),
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
		litertlm.WithBackend("cpu"),
		litertlm.WithMaxTokens(1024),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = client.Close() }()

	sessCfg, err := litertlm.NewSessionConfig()
	if err != nil {
		t.Fatalf("NewSessionConfig: %v", err)
	}
	defer sessCfg.Delete()
	if cfgErr := sessCfg.SetMaxOutputTokens(32); cfgErr != nil {
		t.Fatalf("SetMaxOutputTokens: %v", cfgErr)
	}
	if sampErr := sessCfg.SetSamplerParams(litertlm.SamplerParams{
		Type:        litertlm.SamplerTopP,
		TopK:        40,
		TopP:        0.95,
		Temperature: 0,
		Seed:        1,
	}); sampErr != nil {
		t.Fatalf("SetSamplerParams: %v", sampErr)
	}
	if specErr := sessCfg.SetSpeculativeDecodingConfig(litertlm.SpeculativeDecodingConfig{
		Mode:     litertlm.SpeculativeDecodingOff,
		Lenience: 0,
	}); specErr != nil {
		t.Fatalf("SetSpeculativeDecodingConfig: %v", specErr)
	}

	sess, err := client.Engine().NewSession(sessCfg)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer sess.Delete()

	prefix, err := litertlm.NewTextInputString("The capital of ")
	if err != nil {
		t.Fatalf("NewTextInputString: %v", err)
	}
	defer prefix.Delete()

	if prefillErr := sess.RunPrefill([]litertlm.InputData{prefix}); prefillErr != nil {
		t.Fatalf("RunPrefill prefix: %v", prefillErr)
	}
	if cpErr := sess.SaveCheckpoint("after-prefix"); cpErr != nil {
		t.Fatalf("SaveCheckpoint: %v", cpErr)
	}

	cloned, err := sess.Clone()
	if err != nil {
		t.Fatalf("Session.Clone: %v", err)
	}
	defer cloned.Delete()

	branch1, err := litertlm.NewTextInputString("France is")
	if err != nil {
		t.Fatalf("NewTextInputString branch1: %v", err)
	}
	defer branch1.Delete()
	if b1Err := sess.RunPrefill([]litertlm.InputData{branch1}); b1Err != nil {
		t.Fatalf("RunPrefill branch1: %v", b1Err)
	}

	var out1 strings.Builder
	sawFinal1 := false
	for sc := range sess.RunDecodeAsyncCh() {
		if sc.Err != nil {
			t.Fatalf("RunDecodeAsyncCh branch1: %v", sc.Err)
		}
		out1.WriteString(sc.Text)
		if sc.Final {
			sawFinal1 = true
		}
	}
	if !sawFinal1 || out1.String() == "" {
		t.Fatalf("branch1 output=%q sawFinal=%v", out1.String(), sawFinal1)
	}

	if rwErr := sess.RewindToCheckpoint("after-prefix"); rwErr != nil {
		t.Fatalf("RewindToCheckpoint: %v", rwErr)
	}
	branch2, err := litertlm.NewTextInputString("Japan is")
	if err != nil {
		t.Fatalf("NewTextInputString branch2: %v", err)
	}
	defer branch2.Delete()
	if b2Err := sess.RunPrefill([]litertlm.InputData{branch2}); b2Err != nil {
		t.Fatalf("RunPrefill branch2: %v", b2Err)
	}
	resp2, err := sess.RunDecode()
	if err != nil {
		t.Fatalf("RunDecode branch2: %v", err)
	}
	defer resp2.Delete()
	if resp2.Text(0) == "" {
		t.Fatal("branch2 RunDecode returned empty text")
	}

	if rwsErr := sess.RewindToStep(0); rwsErr != nil {
		t.Fatalf("RewindToStep(0): %v", rwsErr)
	}

	cloneIn, err := litertlm.NewTextInputString("Italy is")
	if err != nil {
		t.Fatalf("NewTextInputString clone: %v", err)
	}
	defer cloneIn.Delete()
	if cPreErr := cloned.RunPrefill([]litertlm.InputData{cloneIn}); cPreErr != nil {
		t.Fatalf("cloned RunPrefill: %v", cPreErr)
	}
	var outClone strings.Builder
	if syncErr := cloned.RunDecodeAsync(func(sc litertlm.StreamChunk) {
		if sc.Err != nil {
			t.Errorf("cloned RunDecodeAsync chunk err: %v", sc.Err)
		}
		outClone.WriteString(sc.Text)
	}); syncErr != nil {
		t.Fatalf("cloned RunDecodeAsync: %v", syncErr)
	}
	if outClone.String() == "" {
		t.Fatal("cloned RunDecodeAsync produced empty text")
	}
}

func TestLive_Phase2_ConstrainedDecoding(t *testing.T) {
	libDir, modelPath := requireTestModel(t)
	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	ctx := context.Background()

	client, err := litertlm.New(ctx,
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
		litertlm.WithBackend("cpu"),
		litertlm.WithMaxTokens(1024),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = client.Close() }()

	chat, err := client.NewChat(ctx,
		litertlm.WithConstrainedDecoding(true),
		litertlm.WithConstraintProvider(litertlm.ConstraintProviderLLGuidance),
	)
	if err != nil {
		if strings.Contains(err.Error(), "Failed to create GemmaModelConstraintProvider") {
			t.Skipf("model %s lacks constraint provider metadata: %v", modelPath, err)
		}
		t.Fatalf("NewChat constrained: %v", err)
	}
	defer func() { _ = chat.Close() }()

	reReply, err := chat.Send(ctx, "Reply with red or blue only.",
		litertlm.WithMaxOutputTokens(8),
		litertlm.WithConstraint(litertlm.ConstraintRegex, `(red|blue)`),
	)
	if err != nil {
		t.Fatalf("regex Send: %v", err)
	}
	gotRe := strings.TrimSpace(reReply.Text())
	if gotRe != "red" && gotRe != "blue" {
		t.Errorf("regex reply = %q, want red or blue", gotRe)
	}

	const schema = `{"type":"object","properties":{"color":{"type":"string"}},"required":["color"],"additionalProperties":false}`
	jsonReply, err := chat.Send(ctx, "Return color green.",
		litertlm.WithMaxOutputTokens(24),
		litertlm.WithConstraint(litertlm.ConstraintJSONSchema, schema),
	)
	if err != nil {
		t.Fatalf("json schema Send: %v", err)
	}
	if !json.Valid([]byte(jsonReply.Text())) {
		t.Errorf("json schema reply is invalid JSON: %q", jsonReply.Text())
	}
	var parsed struct {
		Color string `json:"color"`
	}
	if unmarshalErr := json.Unmarshal([]byte(jsonReply.Text()), &parsed); unmarshalErr != nil {
		t.Errorf("json schema unmarshal failed: %v (raw=%q)", unmarshalErr, jsonReply.Text())
	}
}

func TestLive_Phase2_PenaltiesPromptWaitAndToolChoice(t *testing.T) {
	libDir, modelPath := requireTestModel(t)
	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	ctx := context.Background()

	client, err := litertlm.New(ctx,
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
		litertlm.WithBackend("cpu"),
		litertlm.WithMaxTokens(1024),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = client.Close() }()

	chatPenalties, err := client.NewChat(ctx,
		litertlm.WithPromptTemplate("{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant: "),
	)
	if err != nil {
		t.Fatalf("NewChat prompt template: %v", err)
	}
	defer func() { _ = chatPenalties.Close() }()

	penReply, err := chatPenalties.Send(ctx, "Name two fruits.",
		litertlm.WithMaxOutputTokens(16),
		litertlm.WithRepetitionPenalty(litertlm.RepetitionPenaltyConfig{
			RepetitionPenalty: 1.15,
			PresencePenalty:   0.1,
			FrequencyPenalty:  0.1,
			WindowSize:        32,
		}),
		litertlm.WithNoRepeatNgram(litertlm.NoRepeatNgramConfig{NgramSize: 3, WindowSize: 32}),
		litertlm.WithSuppressTokens(999),
	)
	if err != nil {
		t.Fatalf("penalties Send: %v", err)
	}
	if strings.TrimSpace(penReply.Text()) == "" {
		t.Error("penalties Send returned empty reply")
	}

	if waitErr := chatPenalties.Conversation().WaitUntilDone(); waitErr != nil {
		t.Errorf("WaitUntilDone: %v", waitErr)
	}

	maj, minVer, _, _ := litertlm.APIVersion()
	_, tcErr := chatPenalties.Send(ctx, "Hi.",
		litertlm.WithMaxOutputTokens(4),
		litertlm.WithToolChoice(litertlm.ToolChoiceAuto()),
	)
	if maj == 1 && minVer < 1 && !errors.Is(tcErr, litertlm.ErrUnimplemented) {
		t.Errorf("WithToolChoice on %d.%d = %v, want ErrUnimplemented", maj, minVer, tcErr)
	}
}

func TestLive_Phase2_SingleThreadedStream(t *testing.T) {
	libDir, modelPath := requireTestModel(t)
	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	ctx := context.Background()

	client, err := litertlm.New(ctx,
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
		litertlm.WithBackend("cpu"),
		litertlm.WithMaxTokens(1024),
		litertlm.WithSingleThreadedExecution(true),
	)
	if err != nil {
		t.Fatalf("New single-threaded: %v", err)
	}
	defer func() { _ = client.Close() }()

	chat, err := client.NewChat(ctx)
	if err != nil {
		t.Fatalf("NewChat: %v", err)
	}
	defer func() { _ = chat.Close() }()

	var got strings.Builder
	sawFinal := false
	for ch, streamErr := range chat.SendStream(ctx, "Name one color.", litertlm.WithMaxOutputTokens(12)) {
		if streamErr != nil {
			t.Fatalf("SendStream: %v", streamErr)
		}
		got.WriteString(ch.Text)
		if ch.Final {
			sawFinal = true
		}
	}
	if !sawFinal || strings.TrimSpace(got.String()) == "" {
		t.Errorf("single-threaded SendStream: text=%q sawFinal=%v", got.String(), sawFinal)
	}
}

func TestLive_Phase2_Thinking(t *testing.T) {
	libDir, modelPath := requireTestModel(t)
	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	ctx := context.Background()

	client, err := litertlm.New(ctx,
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
		litertlm.WithBackend("cpu"),
		litertlm.WithMaxTokens(1024),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = client.Close() }()

	chat, err := client.NewChat(ctx,
		litertlm.WithThinking(litertlm.ThinkingConfig{Enabled: true, TokenBudget: 64}),
		litertlm.WithFilterChannelContentFromKVCache(true),
	)
	if err != nil {
		t.Fatalf("NewChat thinking: %v", err)
	}
	defer func() { _ = chat.Close() }()

	r1, err := chat.Send(ctx, "What is 2 + 2?", litertlm.WithMaxOutputTokens(48))
	if err != nil {
		t.Fatalf("turn 1 thinking Send: %v", err)
	}
	if strings.TrimSpace(r1.Text()) == "" {
		t.Error("turn 1 returned empty reply")
	}

	chat2, err := client.NewChat(ctx,
		litertlm.WithThinking(litertlm.ThinkingConfig{Enabled: true, TokenBudget: 64}),
	)
	if err != nil {
		t.Fatalf("NewChat thinking 2: %v", err)
	}
	defer func() { _ = chat2.Close() }()

	r2, err := chat2.Send(ctx, "What is 3 + 3?",
		litertlm.WithMaxOutputTokens(24),
		litertlm.WithTurnThinking(litertlm.ThinkingConfig{Enabled: false}),
	)
	if err != nil {
		t.Fatalf("turn thinking override Send: %v", err)
	}
	if strings.TrimSpace(r2.Text()) == "" {
		t.Error("turn thinking override returned empty reply")
	}
}
