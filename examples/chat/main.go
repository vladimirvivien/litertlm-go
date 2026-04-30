// chat demonstrates the higher-level Conversation API with multi-turn JSON
// messages. Use this with chat-tuned models (Gemma instruct, Llama-Instruct,
// Phi-4, etc.) — the C side automatically applies the model's chat template,
// so the bot output looks like a proper assistant reply.
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

// assistantMessage mirrors the JSON shape SendMessage returns for a chat
// reply: {"role":"assistant","content":[{"type":"text","text":"..."}]}.
// Other content types (image, audio) are surfaced for completeness but
// only `text` items are printed by this example.
type assistantMessage struct {
	Role    string `json:"role"`
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text,omitempty"`
	} `json:"content"`
}

// extractText concatenates every text content part. Returns the parse
// error if the response wasn't valid JSON in the documented shape.
func extractText(jsonResp string) (string, error) {
	var m assistantMessage
	if err := json.Unmarshal([]byte(jsonResp), &m); err != nil {
		return "", fmt.Errorf("unmarshal assistant message: %w", err)
	}
	var b strings.Builder
	for _, p := range m.Content {
		if p.Type == "text" {
			b.WriteString(p.Text)
		}
	}
	return b.String(), nil
}

func main() {
	model := flag.String("model", "", "path to .litertlm model file")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries (falls back to LITERTLM_LIB env)")
	system := flag.String("system", "You are a friendly assistant.", "system message")
	prompt := flag.String("prompt", "", "if set, send this single user message instead of the built-in two-turn demo")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "--model is required")
		os.Exit(2)
	}

	if err := litertlm.Load(*libPath, *backend); err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer litertlm.Close()

	// Silence LiteRT-LM's INFO/WARN chatter. Drop to LogInfo to see it.
	litertlm.SetMinLogLevel(litertlm.LogError)

	settings, err := litertlm.NewEngineSettings(*model, *backend, nil, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "settings: %v\n", err)
		os.Exit(1)
	}
	defer settings.Delete()

	engine, err := litertlm.NewEngine(settings)
	if err != nil {
		fmt.Fprintf(os.Stderr, "engine: %v\n", err)
		os.Exit(1)
	}
	defer engine.Delete()

	// systemMessageJSON expects just the content (string or content array),
	// not a full {role,content} envelope — the C side wraps it in a system
	// message itself (see c/engine.cc:litert_lm_conversation_create). A
	// JSON-encoded string is parsed as a JSON value and used as content
	// directly; passing the envelope makes the chat template silently drop
	// the system prompt.
	sysJSON, _ := json.Marshal(*system)
	cfg, err := litertlm.NewConversationConfig(engine, 0, string(sysJSON), "", "", false)
	if err != nil {
		fmt.Fprintf(os.Stderr, "conv cfg: %v\n", err)
		os.Exit(1)
	}
	defer cfg.Delete()

	conv, err := engine.NewConversation(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "conv: %v\n", err)
		os.Exit(1)
	}
	defer conv.Delete()

	turns := []string{
		"Hi, what is your name?",
		"Tell me a one-sentence fun fact about octopuses.",
	}
	if *prompt != "" {
		turns = []string{*prompt}
	}

	for _, msg := range turns {
		msgJSON, _ := json.Marshal(map[string]string{"role": "user", "content": msg})
		fmt.Printf("user> %s\n", msg)

		resp, err := conv.SendMessage(string(msgJSON), "")
		if err != nil {
			fmt.Fprintf(os.Stderr, "send: %v\n", err)
			os.Exit(1)
		}
		text, err := extractText(resp)
		if err != nil {
			fmt.Fprintf(os.Stderr, "parse reply: %v\n  raw: %s\n", err, resp)
			os.Exit(1)
		}
		fmt.Printf("bot>  %s\n\n", text)
	}
}
