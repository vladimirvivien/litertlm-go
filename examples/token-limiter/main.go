// token-limiter demonstrates how to dynamically restrict token generation length
// on a per-call basis using the WithMaxOutputTokens runtime option.
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
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries")
	prompt := flag.String("prompt", "Write a long list of 20 historical cities.", "prompt text")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "--model is required")
		os.Exit(2)
	}

	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	ctx := context.Background()

	client, err := litertlm.New(ctx,
		litertlm.WithLib(*libPath),
		litertlm.WithModel(*model),
		litertlm.WithBackend(*backend),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	// 1. Run 1: Constrained Output (Max 8 tokens)
	fmt.Printf("--- Run 1: Constrained Output (WithMaxOutputTokens: 8) ---\n")
	chat1, err := client.NewChat(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new chat: %v\n", err)
		os.Exit(1)
	}
	reply1, err := chat1.Send(ctx, *prompt, litertlm.WithMaxOutputTokens(8))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Send failed: %v\n", err)
		chat1.Close()
		os.Exit(1)
	}
	fmt.Printf("Output 1:\n%s\n\n", reply1.Text())
	chat1.Close()

	// 2. Run 2: Full Output (Max 128 tokens)
	fmt.Printf("--- Run 2: Full Output (WithMaxOutputTokens: 128) ---\n")
	chat2, err := client.NewChat(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new chat: %v\n", err)
		os.Exit(1)
	}
	reply2, err := chat2.Send(ctx, *prompt, litertlm.WithMaxOutputTokens(128))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Send failed: %v\n", err)
		chat2.Close()
		os.Exit(1)
	}
	fmt.Printf("Output 2:\n%s\n", reply2.Text())
	chat2.Close()
}
