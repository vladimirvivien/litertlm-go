// clone demonstrates Chat.Clone: open a Chat, prefill some shared
// context with one turn, branch the conversation into two independent
// clones, and send a different follow-up to each. The two replies
// share everything before the branch point and diverge after.
//
// Useful pattern for branching tool loops (run N candidate tools off
// one prefilled prompt), for structured-output retries that should
// start from identical state, and for A/B-style decoding comparisons.
//
// See README.md in this directory for prerequisites, the upstream
// support matrix, and current limitations.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	model := flag.String("model", "", "path to .litertlm model file (required)")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.16.0)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	cacheDir := flag.String("cache-dir", "", "directory passed to WithCacheDir; empty leaves the engine default")
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

	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	opts := []litertlm.Option{
		litertlm.WithLib(resolvedLib),
		litertlm.WithModel(resolvedModel),
		litertlm.WithBackend(*backend),
	}
	if *cacheDir != "" {
		opts = append(opts, litertlm.WithCacheDir(*cacheDir))
	}

	c, err := litertlm.New(ctx, opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer c.Close()

	chat, err := c.NewChat(ctx, litertlm.WithSystemPrompt(
		"You are a concise assistant. Answer each question in one short sentence."))
	if err != nil {
		fmt.Fprintf(os.Stderr, "new chat: %v\n", err)
		os.Exit(1)
	}
	defer chat.Close()

	// Prefill some shared context — both branches will inherit this turn.
	setup, err := chat.Send(ctx, "Remember: my pet's name is Comet and he is a corgi.")
	if err != nil {
		fmt.Fprintf(os.Stderr, "setup turn: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("== setup turn ==")
	fmt.Println("assistant:", setup.Text())
	fmt.Println()

	clone, err := chat.Clone()
	if err != nil {
		fmt.Println("Chat.Clone failed:", err)
		fmt.Println()
		fmt.Println("As of LiteRT-LM v0.12.0 the LiteRT executor used by Gemma 4")
		fmt.Println("on CPU and GPU returns Unimplemented from Session.Clone.")
		fmt.Println("The wrapper binds the C symbol correctly.")
		// Upstream-side gap, not user-side — exit 0.
		return
	}
	defer clone.Close()

	// Branch A on the original Chat.
	branchA, err := chat.Send(ctx, "What kind of dog is Comet?")
	if err != nil {
		fmt.Fprintf(os.Stderr, "branch A: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("== branch A (original) ==")
	fmt.Println("user: What kind of dog is Comet?")
	fmt.Println("assistant:", branchA.Text())
	fmt.Println()

	// Branch B on the clone. Same prefilled history, different follow-up.
	branchB, err := clone.Send(ctx, "Suggest a fun nickname for Comet.")
	if err != nil {
		fmt.Fprintf(os.Stderr, "branch B: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("== branch B (clone) ==")
	fmt.Println("user: Suggest a fun nickname for Comet.")
	fmt.Println("assistant:", branchB.Text())
}
