// cancel demonstrates aborting an in-flight streaming generation
// using the high-level Client.GenerateStream API and context
// cancellation. After N chunks have been received, we call cancel()
// on the context; wireCancel inside Client.GenerateStream sees
// ctx.Done() fire and calls Session.Cancel() under the hood.
//
// See README.md in this directory for prerequisites and usage.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	model := flag.String("model", "", "path to .litertlm model file")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.16.0)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries (falls back to LITERTLM_LIB env)")
	prompt := flag.String("prompt", "Tell me a long story about a dragon and a wizard.", "prompt text")
	cancelAfter := flag.Int("cancel-after", 8, "number of chunks to receive before cancelling")
	maxTokens := flag.Int("max", 4096, "max total tokens (prompt + output)")
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

	rootCtx := context.Background()
	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	client, err := litertlm.New(rootCtx,
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

	ctx, cancel := context.WithCancel(rootCtx)
	defer cancel()

	fmt.Printf("prompt:  %s\n", *prompt)
	fmt.Println("output:")

	start := time.Now()
	count := 0
	for chunk, err := range client.GenerateStream(ctx, *prompt) {
		if err != nil {
			fmt.Fprintf(os.Stderr, "\nstream error: %v\n", err)
			break
		}
		fmt.Print(chunk.Text)
		count++
		if count == *cancelAfter && !chunk.Final {
			fmt.Printf("\n\n[%v] cancelling after %d chunks ...\n",
				time.Since(start).Round(time.Millisecond), count)
			cancel()
		}
		if chunk.Final {
			fmt.Println()
		}
	}
	fmt.Printf("done (%d chunks total, wall=%v)\n",
		count, time.Since(start).Round(time.Millisecond))
}
