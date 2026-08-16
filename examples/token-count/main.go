// token-count demonstrates Chat.TokenCount: the running number of
// tokens held in the conversation's KV cache, read after each turn to
// project a chat against the engine's max-token budget. It needs no
// benchmark collection — TokenCount reads the live KV-cache size
// directly (LiteRT-LM v0.13.1+).
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
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.16.0)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries (falls back to LITERTLM_LIB env)")
	maxTokens := flag.Int("max", 4096, "engine max tokens (prompt + output); TokenCount is reported as a fraction of this")
	flag.Parse()

	ctx := context.Background()

	resolvedLib := *libPath
	if *getLib != "" {
		staged, err := litertlm.LibFetch("", "", *getLib)
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
		litertlm.WithMaxTokens(*maxTokens),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	chat, err := client.NewChat(ctx, litertlm.WithSystemPrompt("You are a concise assistant."))
	if err != nil {
		fmt.Fprintf(os.Stderr, "new chat: %v\n", err)
		os.Exit(1)
	}
	defer chat.Close()

	report := func(label string) {
		n, err := chat.TokenCount()
		if err != nil {
			fmt.Fprintf(os.Stderr, "token count: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("    [%s] KV cache: %d / %d tokens (%.1f%% of budget)\n",
			label, n, *maxTokens, 100*float64(n)/float64(*maxTokens))
	}

	report("start")
	for _, msg := range []string{
		"Name three primary colors.",
		"Now name three secondary colors.",
		"Summarize this conversation in one sentence.",
	} {
		fmt.Printf("user> %s\n", msg)
		reply, err := chat.Send(ctx, msg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "send: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("bot>  %s\n", reply.Text())
		report("after turn")
		fmt.Println()
	}
}
