package litertlm_test

import (
	"context"
	"encoding/json"
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

// requireTestModel returns lib-dir and model paths from
// LITERTLM_TEST_LIB and LITERTLM_TEST_MODEL. Calls t.Skip when
// testing.Short() is set or when either env var is empty.
// Integration tests that need a real engine call this once at entry.
func requireTestModel(t *testing.T) (libDir, modelPath string) {
	t.Helper()
	if testing.Short() {
		t.Skip("integration test skipped in -short mode")
	}
	libDir = os.Getenv("LITERTLM_TEST_LIB")
	modelPath = os.Getenv("LITERTLM_TEST_MODEL")
	if libDir == "" || modelPath == "" {
		t.Skip("integration test requires LITERTLM_TEST_LIB and LITERTLM_TEST_MODEL")
	}
	return libDir, modelPath
}

// buildIntegrationConversation constructs a Conversation against a
// real model and registers t.Cleanup for every C-side handle so test
// ordering does not leak memory. Backend is CPU.
func buildIntegrationConversation(t *testing.T) litertlm.Conversation {
	t.Helper()
	libDir, modelPath := requireTestModel(t)

	if err := litertlm.Load(libDir, "cpu", ""); err != nil {
		t.Fatalf("Load: %v", err)
	}
	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	settings, err := litertlm.NewEngineSettings(modelPath, "cpu", nil, nil)
	if err != nil {
		t.Fatalf("NewEngineSettings: %v", err)
	}
	t.Cleanup(func() { settings.Delete() })
	settings.SetMaxNumTokens(1024)

	engine, err := litertlm.NewEngine(settings)
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	t.Cleanup(func() { engine.Delete() })

	sessCfg, err := litertlm.NewSessionConfig()
	if err != nil {
		t.Fatalf("NewSessionConfig: %v", err)
	}
	t.Cleanup(func() { sessCfg.Delete() })
	sessCfg.SetMaxOutputTokens(128)
	sessCfg.SetApplyPromptTemplate(true)

	systemMessageJSON, err := json.Marshal("You are a concise assistant.")
	if err != nil {
		t.Fatalf("marshal system: %v", err)
	}
	convCfg, err := litertlm.NewConversationConfig(
		engine, sessCfg,
		string(systemMessageJSON),
		"", "", false,
	)
	if err != nil {
		t.Fatalf("NewConversationConfig: %v", err)
	}
	t.Cleanup(func() { convCfg.Delete() })

	conv, err := engine.NewConversation(convCfg)
	if err != nil {
		t.Fatalf("NewConversation: %v", err)
	}
	t.Cleanup(func() { conv.Delete() })
	return conv
}

// userMessageJSON returns the {"role":"user","content":"..."} envelope
// the chat template expects.
func userMessageJSON(t *testing.T, text string) string {
	t.Helper()
	b, err := json.Marshal(map[string]string{"role": "user", "content": text})
	if err != nil {
		t.Fatalf("marshal user message: %v", err)
	}
	return string(b)
}

// RenderMessage runs the chat template only — no prefill, no decode —
// against the supplied message JSON. The result is non-empty and
// contains the user's text body.
func TestConversation_RenderMessage_NonEmpty(t *testing.T) {
	conv := buildIntegrationConversation(t)
	msg := userMessageJSON(t, "Hello, world!")

	rendered, err := conv.RenderMessage(msg)
	if err != nil {
		t.Fatalf("RenderMessage: %v", err)
	}
	if rendered == "" {
		t.Fatal("RenderMessage returned empty string")
	}
	if !strings.Contains(rendered, "Hello, world!") {
		t.Errorf("rendered output missing user text; got: %q", rendered)
	}
}

func TestConversation_RenderPreface(t *testing.T) {
	conv := buildIntegrationConversation(t)

	rendered, err := conv.RenderPreface()
	if err != nil {
		t.Fatalf("RenderPreface: %v", err)
	}
	if rendered == "" {
		t.Fatal("RenderPreface returned empty string")
	}
	t.Logf("Rendered preface: %q", rendered)
}

// Re-rendering the same JSON produces byte-identical output. The
// template is deterministic and stateless across calls.
func TestConversation_RenderMessage_Deterministic(t *testing.T) {
	conv := buildIntegrationConversation(t)
	msg := userMessageJSON(t, "Deterministic check.")

	first, err := conv.RenderMessage(msg)
	if err != nil {
		t.Fatalf("first RenderMessage: %v", err)
	}
	second, err := conv.RenderMessage(msg)
	if err != nil {
		t.Fatalf("second RenderMessage: %v", err)
	}
	if first != second {
		t.Errorf("RenderMessage not deterministic:\n  first=%q\n  second=%q", first, second)
	}
}

// Different roles produce different renderings — the chat template
// must distinguish user from assistant turns.
func TestConversation_RenderMessage_RoleAffectsOutput(t *testing.T) {
	conv := buildIntegrationConversation(t)
	body := "the same content body"

	userMsg, _ := json.Marshal(map[string]string{"role": "user", "content": body})
	assistantMsg, _ := json.Marshal(map[string]string{"role": "assistant", "content": body})

	userRendered, err := conv.RenderMessage(string(userMsg))
	if err != nil {
		t.Fatalf("user RenderMessage: %v", err)
	}
	assistantRendered, err := conv.RenderMessage(string(assistantMsg))
	if err != nil {
		t.Fatalf("assistant RenderMessage: %v", err)
	}
	if userRendered == assistantRendered {
		t.Errorf("user and assistant renderings identical; template not honoring role:\n  %q", userRendered)
	}
}

// Calling RenderMessage on a zero-value Conversation returns the
// documented invalid-conversation error without invoking the C side.
func TestConversation_RenderMessage_InvalidConversation(t *testing.T) {
	var conv litertlm.Conversation
	_, err := conv.RenderMessage(`{"role":"user","content":"hi"}`)
	if err == nil {
		t.Fatal("expected error from zero-value Conversation")
	}
	if !strings.Contains(err.Error(), "invalid conversation") {
		t.Errorf("error should name the invalid-conversation case; got: %v", err)
	}
}

// TokenCount returns a positive KV-cache token count after a turn and
// grows with conversation length. The integration harness does not
// enable benchmark collection, so this also confirms TokenCount works
// without it (unlike the per-turn BenchmarkInfo counts).
func TestConversation_TokenCount(t *testing.T) {
	conv := buildIntegrationConversation(t)

	before, err := conv.TokenCount()
	if err != nil {
		t.Fatalf("TokenCount before: %v", err)
	}

	if _, sendErr := conv.SendMessage(userMessageJSON(t, "Say hello in one word."), "", litertlm.OptionalArgs(0)); sendErr != nil {
		t.Fatalf("SendMessage: %v", sendErr)
	}

	after, err := conv.TokenCount()
	if err != nil {
		t.Fatalf("TokenCount after: %v", err)
	}
	t.Logf("KV-cache token count: before=%d after=%d", before, after)
	if after <= 0 {
		t.Fatalf("expected a positive token count after a turn, got %d", after)
	}
	if after <= before {
		t.Errorf("token count did not grow after a turn: before=%d after=%d", before, after)
	}
}

// A zero-value Conversation reports the invalid-conversation error
// rather than calling the C side.
func TestConversation_TokenCount_Invalid(t *testing.T) {
	var conv litertlm.Conversation
	if _, err := conv.TokenCount(); err == nil {
		t.Fatal("expected error from zero-value Conversation")
	}
}

func TestClient_WithModelFd(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("raw file descriptor model loading is unsupported on Windows due to CRT boundary limits")
	}
	libDir, modelPath := requireTestModel(t)

	if err := litertlm.Load(libDir, "cpu", ""); err != nil {
		t.Fatalf("Load: %v", err)
	}

	file, err := os.Open(modelPath)
	if err != nil {
		t.Fatalf("failed to open model file: %v", err)
	}

	client, err := litertlm.New(context.Background(),
		litertlm.WithLib(libDir),
		litertlm.WithModelFd(int(file.Fd())),
	)
	if err != nil {
		_ = file.Close()
		t.Fatalf("New with FD: %v", err)
	}
	defer func() { _ = client.Close() }()

	// Simple generation check to verify it loaded correctly.
	res, err := client.Generate(context.Background(), "Paris is the capital of")
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if !strings.Contains(strings.ToLower(res), "france") {
		t.Logf("expected answer to contain 'france', got: %q", res)
	}
}

func TestClient_WithThreads(t *testing.T) {
	libDir, modelPath := requireTestModel(t)

	if err := litertlm.Load(libDir, "cpu", ""); err != nil {
		t.Fatalf("Load: %v", err)
	}

	client, err := litertlm.New(context.Background(),
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
		litertlm.WithNumThreads(2),
	)
	if err != nil {
		t.Fatalf("New with threads: %v", err)
	}
	defer func() { _ = client.Close() }()

	// Simple generation check to verify it loaded correctly.
	res, err := client.Generate(context.Background(), "Paris is the capital of")
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if !strings.Contains(strings.ToLower(res), "france") {
		t.Logf("expected answer to contain 'france', got: %q", res)
	}
}

func TestEngineSettings_Options(t *testing.T) {
	libDir, modelPath := requireTestModel(t)

	if err := litertlm.Load(libDir, "cpu", ""); err != nil {
		t.Fatalf("Load: %v", err)
	}

	settings, err := litertlm.NewEngineSettings(modelPath, "cpu", nil, nil)
	if err != nil {
		t.Fatalf("NewEngineSettings: %v", err)
	}
	defer settings.Delete()

	// Test threading setters
	settings.SetNumThreads(4)
	settings.SetAudioNumThreads(2)

	// Test LoRA setters
	settings.SetLoraRank(8)
	settings.SetAudioLoraRank(8)

	// These might return errors on CPU or on models without audio/vision components.
	// We check that they call the FFI layer safely without panics.
	_ = settings.SetSupportedLoraRanks([]int{8, 16})
	_ = settings.SetSupportedAudioLoraRanks([]int{8, 16})
}


