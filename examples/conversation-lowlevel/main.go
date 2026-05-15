// conversation-lowlevel runs a two-turn chat directly against the
// low-level Conversation API: it constructs a SessionConfig,
// ConversationConfig and Conversation by hand — the same path
// Client.NewChat / Chat.Send build internally.
//
// See README.md in this directory for prerequisites and usage.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

const defaultSystemPrompt = "You are a concise assistant. Answer in one sentence."

// defaultExtraContext must be a JSON object — non-objects are
// silently dropped by the C side. The chat template merges the
// object's top-level keys into the conversation preface.
const defaultExtraContext = `{
  "notes": [
    "The current user is identified only as \"the user\".",
    "Today's date is unspecified."
  ]
}`

func main() {
	model := flag.String("model", "", "path to .litertlm model file (required)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	systemPrompt := flag.String("system", defaultSystemPrompt, "system prompt (bare content; gets JSON-encoded into the conversation config)")
	extraContext := flag.String("extra-context", defaultExtraContext, "extra-context preface attached to the conversation; empty to skip")
	turn1 := flag.String("turn1", "My name is Vlad. Tell me one fun fact about the Go programming language.", "first user message")
	turn2 := flag.String("turn2", "What was my name again?", "second user message; relies on turn 1 still in the KV cache")
	maxTokens := flag.Int("max-tokens", 2048, "engine total token budget (prompt + output)")
	maxOutputTokens := flag.Int("max-output-tokens", 256, "per-turn decode cap (SessionConfig.SetMaxOutputTokens)")
	topP := flag.Float64("top-p", 0.95, "nucleus sampling top_p (used when -temp > 0)")
	temperature := flag.Float64("temp", 0.0, "sampler temperature; 0 skips SetSamplerParams (C engine default sampler is used)")
	seed := flag.Int("seed", 1, "sampler seed (top-p/top-k only)")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "--model is required")
		os.Exit(2)
	}

	if err := litertlm.Load(*libPath, *backend, ""); err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer litertlm.Close()
	litertlm.SetMinLogLevel(litertlm.LogError)

	// ---- Engine ---------------------------------------------------------
	settings, err := litertlm.NewEngineSettings(*model, *backend, nil, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "settings: %v\n", err)
		os.Exit(1)
	}
	defer settings.Delete()
	settings.SetMaxNumTokens(*maxTokens)
	settings.EnableBenchmark()

	engine, err := litertlm.NewEngine(settings)
	if err != nil {
		fmt.Fprintf(os.Stderr, "engine: %v\n", err)
		os.Exit(1)
	}
	defer engine.Delete()

	// ---- SessionConfig --------------------------------------------------
	sessCfg, err := litertlm.NewSessionConfig()
	if err != nil {
		fmt.Fprintf(os.Stderr, "session config: %v\n", err)
		os.Exit(1)
	}
	// NewConversationConfig copies the SessionConfig contents on the C
	// side, so the local handle is safe to release as soon as the
	// ConversationConfig has been built.
	defer sessCfg.Delete()

	// SamplerGreedy (type 3) is not implemented at the C side and
	// fails conversation_create. Skip SetSamplerParams when the caller
	// did not opt into an explicit sampler.
	if sampler, ok := buildSampler(*temperature, *topP, *seed); ok {
		sessCfg.SetSamplerParams(sampler)
	}
	sessCfg.SetMaxOutputTokens(*maxOutputTokens)
	// Required for multi-turn chat: wraps each turn in the model's
	// start/end markers. With it off, the Conversation degenerates
	// into raw completion.
	sessCfg.SetApplyPromptTemplate(true)

	// ---- ConversationConfig --------------------------------------------
	// systemMessage is bare content (a JSON-encoded string), not a
	// {role,content} envelope — the C side wraps it itself. Passing
	// the envelope makes the chat template drop the system message.
	systemMessageJSON, err := json.Marshal(*systemPrompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "marshal system: %v\n", err)
		os.Exit(1)
	}

	convCfg, err := litertlm.NewConversationConfig(
		engine, sessCfg,
		string(systemMessageJSON),
		"",    // toolsJSON
		"",    // messagesJSON
		false, // enableConstrainedDecoding
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "conversation config: %v\n", err)
		os.Exit(1)
	}
	defer convCfg.Delete()

	if *extraContext != "" {
		if err = convCfg.SetExtraContext(*extraContext); err != nil {
			fmt.Fprintf(os.Stderr, "extra context: %v\n", err)
			os.Exit(1)
		}
	}

	conv, err := engine.NewConversation(convCfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "conversation: %v\n", err)
		os.Exit(1)
	}
	defer conv.Delete()

	// ---- Turn 1 ---------------------------------------------------------
	turn1JSON := encodeUserMessage(*turn1)

	// RenderMessage runs the chat template only — no prefill, no
	// decode — and returns the string the model would be fed.
	rendered, err := conv.RenderMessage(turn1JSON)
	if err != nil {
		fmt.Fprintf(os.Stderr, "render: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("=== Rendered turn 1 (what the model actually sees) ===")
	fmt.Println(strings.TrimSpace(rendered))

	fmt.Printf("\nuser>      %s\n", *turn1)
	reply1, err := conv.SendMessage(turn1JSON, "")
	if err != nil {
		fmt.Fprintf(os.Stderr, "send turn 1: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("assistant> %s\n", textFromReply(reply1))

	// ---- Turn 2 ---------------------------------------------------------
	// Same Conversation handle: the KV cache from turn 1 persists.
	turn2JSON := encodeUserMessage(*turn2)
	fmt.Printf("\nuser>      %s\n", *turn2)
	reply2, err := conv.SendMessage(turn2JSON, "")
	if err != nil {
		fmt.Fprintf(os.Stderr, "send turn 2: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("assistant> %s\n", textFromReply(reply2))

	// ---- Benchmark ------------------------------------------------------
	bi, err := conv.BenchmarkInfo()
	if err != nil {
		fmt.Fprintf(os.Stderr, "benchmark info: %v\n", err)
		os.Exit(1)
	}
	defer bi.Delete()

	fmt.Println("\n=== Conversation BenchmarkInfo ===")
	fmt.Printf("time to first token: %.3fs\n", bi.TimeToFirstToken())
	fmt.Printf("prefill turns:       %d\n", bi.NumPrefillTurns())
	fmt.Printf("decode  turns:       %d\n", bi.NumDecodeTurns())
	for i := 0; i < bi.NumDecodeTurns(); i++ {
		fmt.Printf("  turn %d: prefill=%d tok @ %.1f tok/s | decode=%d tok @ %.1f tok/s\n",
			i,
			bi.PrefillTokenCount(i), bi.PrefillTokensPerSec(i),
			bi.DecodeTokenCount(i), bi.DecodeTokensPerSec(i),
		)
	}
}

// encodeUserMessage builds the {role,content} envelope SendMessage
// expects.
func encodeUserMessage(text string) string {
	b, err := json.Marshal(map[string]string{"role": "user", "content": text})
	if err != nil {
		panic(err)
	}
	return string(b)
}

// textFromReply extracts the assistant text from the JSON envelope
// SendMessage returns:
// `{"role":"assistant","content":[{"type":"text","text":"..."}]}`.
func textFromReply(raw string) string {
	var msg struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		return strings.TrimSpace(raw)
	}
	var b strings.Builder
	for _, p := range msg.Content {
		if p.Type == "text" {
			b.WriteString(p.Text)
		}
	}
	return strings.TrimSpace(b.String())
}

// buildSampler returns a SamplerParams shaped by the CLI flags and a
// bool that is false when no override is requested (temp == 0). On
// false, callers must skip SetSamplerParams: the C side does not
// implement SamplerGreedy (type 3) and would fail
// conversation_create.
func buildSampler(temp, topP float64, seed int) (litertlm.SamplerParams, bool) {
	if temp <= 0 {
		return litertlm.SamplerParams{}, false
	}
	return litertlm.SamplerParams{
		Type:        litertlm.SamplerTopP,
		TopK:        40,
		TopP:        float32(topP),
		Temperature: float32(temp),
		Seed:        int32(seed),
	}, true
}
