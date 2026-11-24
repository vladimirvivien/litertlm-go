// thinking demonstrates session-level and per-turn reasoning controls
// using WithThinking, WithTurnThinking, and WithFilterChannelContentFromKVCache.
// Works best on reasoning models such as Qwen3 or Gemma 4.
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
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.19.1)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	budget := flag.Int("budget", 128, "thinking token budget for turn 1 (-1 for unlimited)")
	maxOutput := flag.Int("max-output", 192, "max output tokens per turn")
	flag.Parse()

	ctx := context.Background()
	resolvedLib := *libPath
	if *getLib != "" {
		staged, err := litertlm.FetchLib(ctx, "", "", *getLib)
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

	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	client, err := litertlm.New(ctx,
		litertlm.WithLib(resolvedLib),
		litertlm.WithModel(resolvedModel),
		litertlm.WithBackend(*backend),
		litertlm.WithMaxTokens(2048),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = client.Close() }()

	chat, err := client.NewChat(ctx,
		litertlm.WithThinking(litertlm.ThinkingConfig{
			Enabled:     true,
			TokenBudget: *budget,
		}),
		litertlm.WithFilterChannelContentFromKVCache(true),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new chat: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = chat.Close() }()

	fmt.Println("=== Turn 1 (thinking enabled) ===")
	reply1, err := chat.Send(ctx,
		"A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?",
		litertlm.WithMaxOutputTokens(*maxOutput),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "turn 1: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(reply1.Text())

	fmt.Println("=== Turn 2 (thinking disabled via WithTurnThinking) ===")
	chat2, err := client.NewChat(ctx,
		litertlm.WithThinking(litertlm.ThinkingConfig{Enabled: true, TokenBudget: *budget}),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new chat 2: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = chat2.Close() }()

	reply2, err := chat2.Send(ctx,
		"What is 2 + 2? Answer with the number only.",
		litertlm.WithMaxOutputTokens(64),
		litertlm.WithTurnThinking(litertlm.ThinkingConfig{Enabled: false}),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "turn 2: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(reply2.Text())
}
