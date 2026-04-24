// hello demonstrates a minimal synchronous inference with litertlm-go using
// the low-level Session API.
//
// See README.md in this directory for prerequisites and usage.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	model := flag.String("model", "", "path to .litertlm model file")
	prompt := flag.String("prompt", "The capital of France is", "prompt text")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries (falls back to LITERTLM_LIB env)")
	maxTokens := flag.Int("max", 1024, "max total tokens (prompt + output); must be >= the model's smallest prefill signature, typically 128")
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
	settings.SetMaxNumTokens(*maxTokens)

	engine, err := litertlm.NewEngine(settings)
	if err != nil {
		fmt.Fprintf(os.Stderr, "engine: %v\n", err)
		os.Exit(1)
	}
	defer engine.Delete()

	session, err := engine.NewSession(0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "session: %v\n", err)
		os.Exit(1)
	}
	defer session.Delete()

	resp, err := session.GenerateContent([]litertlm.InputData{
		litertlm.NewTextInputString(*prompt),
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}
	defer resp.Delete()

	n := resp.NumCandidates()
	if n == 0 {
		fmt.Fprintln(os.Stderr, "no candidates returned")
		os.Exit(1)
	}

	emptyAll := true
	for i := 0; i < n; i++ {
		text := resp.Text(i)
		fmt.Printf("[%d] %s\n", i, text)
		if text != "" {
			emptyAll = false
		}
	}

	if emptyAll {
		fmt.Fprintln(os.Stderr, `
hint: the model returned an empty completion. This typically means the
prompt was sent to a chat-tuned model without its chat template, so the
model produced an end-of-sequence token immediately. Either:
  - try a "completion-style" prompt that the model can extend (the default
    "The capital of France is" works on Gemma 4 base + many models), or
  - run the chat example, which uses the Conversation API and applies the
    model's chat template automatically:
      go run ./examples/chat -model `+*model)
	}
}
