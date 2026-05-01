// chat demonstrates multi-turn conversation using the high-level
// Client.NewChat / Chat.Send API. The C side applies the model's chat
// template under the hood, so the bot output looks like a proper
// assistant reply.
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
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries (falls back to LITERTLM_LIB env)")
	system := flag.String("system", "You are a friendly assistant.", "system message")
	prompt := flag.String("prompt", "", "if set, send this single user message instead of the built-in two-turn demo")
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
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	chat, err := client.NewChat(ctx, litertlm.WithSystemPrompt(*system))
	if err != nil {
		fmt.Fprintf(os.Stderr, "new chat: %v\n", err)
		os.Exit(1)
	}
	defer chat.Close()

	turns := []string{
		"Hi, what is your name?",
		"Tell me a one-sentence fun fact about octopuses.",
	}
	if *prompt != "" {
		turns = []string{*prompt}
	}

	for _, msg := range turns {
		fmt.Printf("user> %s\n", msg)
		reply, err := chat.Send(ctx, msg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "send: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("bot>  %s\n\n", reply.Text())
	}
}
