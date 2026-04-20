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

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

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

	if err := litertlm.Load(*libPath); err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer litertlm.Close()

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

	sysJSON, _ := json.Marshal(map[string]string{"role": "system", "content": *system})
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
		fmt.Printf("bot>  %s\n\n", resp)
	}
}
