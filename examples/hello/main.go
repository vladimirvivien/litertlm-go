// hello demonstrates a minimal synchronous inference using the
// high-level Client API. For the equivalent low-level (Session-based)
// flow, see examples/prefill-decode.
//
// See README.md in this directory for prerequisites and usage.
package main

import (
	"context"
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

	ctx := context.Background()
	client, err := litertlm.New(ctx,
		litertlm.WithLib(*libPath),
		litertlm.WithModel(*model),
		litertlm.WithBackend(*backend),
		litertlm.WithMaxTokens(*maxTokens),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	text, err := client.Generate(ctx, *prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(text)

	if text == "" {
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
