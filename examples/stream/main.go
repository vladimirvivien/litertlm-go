// stream demonstrates token-by-token streaming on the high-level
// Chat.SendStream API. The C side applies the model's chat template
// before each turn, so chat-tuned models (Gemma 4, etc.) produce a
// proper assistant reply instead of degenerating on a bare instruction
// prompt.
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
	model := flag.String("model", "", "path to .litertlm model file (falls back to LITERTLM_MODEL env)")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	prompt := flag.String("prompt", "Write a short haiku about the sea.", "prompt text")
	system := flag.String("system", "You are a friendly assistant.", "system message")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries (falls back to LITERTLM_LIB env)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.16.0)")
	maxTokens := flag.Int("max", 1024, "max total tokens (prompt + output); must be >= the model's smallest prefill signature, typically 128")
	speculative := flag.Bool("speculative", false, "enable multi-token-prediction (MTP) speculative decoding; requires a model with an MTP draft head (e.g. litert-community/gemma-4-E4B)")
	flag.Parse()

	ctx := context.Background()

	resolvedLib := *libPath
	if *getLib != "" {
		staged, err := litertlm.FetchLib("", "", *getLib)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fetch library: %v\n", err)
			os.Exit(1)
		}
		resolvedLib = staged
	}

	resolvedModel := *model
	if resolvedModel == "" {
		resolvedModel = os.Getenv("LITERTLM_MODEL")
	}
	if *getModel != "" {
		staged, err := litertlm.FetchModel(ctx, *getModel)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fetch model: %v\n", err)
			os.Exit(1)
		}
		resolvedModel = staged
	}

	if resolvedModel == "" {
		fmt.Fprintln(os.Stderr, "--model or --get-model (or LITERTLM_MODEL env) is required")
		os.Exit(2)
	}

	opts := []litertlm.Option{
		litertlm.WithLib(resolvedLib),
		litertlm.WithModel(resolvedModel),
		litertlm.WithBackend(*backend),
		litertlm.WithMaxTokens(*maxTokens),
	}
	if *speculative {
		opts = append(opts, litertlm.WithSpeculativeDecodingEnabled(true))
	}

	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	client, err := litertlm.New(ctx, opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = client.Close() }()

	chat, err := client.NewChat(ctx, litertlm.WithSystemPrompt(*system))
	if err != nil {
		fmt.Fprintf(os.Stderr, "new chat: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = chat.Close() }()

	fmt.Printf("user> %s\nbot>  ", *prompt)
	for chunk, err := range chat.SendStream(ctx, *prompt) {
		if err != nil {
			fmt.Fprintf(os.Stderr, "\nstream error: %v\n", err)
			os.Exit(1)
		}
		fmt.Print(chunk.Text)
		if chunk.Final {
			fmt.Println()
		}
	}
}
