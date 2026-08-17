// tokenize shows the high-level Client.Tokenize / Client.TokenLength
// helpers paired with the underlying Engine accessors (Detokenize,
// StartTokenIDs, StopTokenIDs) reached via Client.Engine(). No
// inference is run.
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
	model := flag.String("model", "", "path to .litertlm model file (required)")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.16.0)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	text := flag.String("text", "Hello, world. How are you today?", "text to tokenize")
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

	client, err := litertlm.New(ctx,
		litertlm.WithLib(resolvedLib),
		litertlm.WithModel(resolvedModel),
		litertlm.WithBackend(*backend),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	tokens, err := client.Tokenize(*text)
	if err != nil {
		fmt.Fprintf(os.Stderr, "tokenize: %v\n", err)
		os.Exit(1)
	}
	length, err := client.TokenLength(*text)
	if err != nil {
		fmt.Fprintf(os.Stderr, "token length: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("text:         %q\n", *text)
	fmt.Printf("Client.Tokenize    (%d): %v\n", len(tokens), tokens)
	fmt.Printf("Client.TokenLength (%d)\n", length)

	engine := client.Engine()

	roundTrip, err := engine.Detokenize(tokens)
	if err != nil {
		fmt.Fprintf(os.Stderr, "detokenize: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Engine.Detokenize:    %q\n", roundTrip)

	start, err := engine.StartTokenIDs()
	if err != nil {
		fmt.Fprintf(os.Stderr, "start tokens: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Engine.StartTokenIDs: %v\n", start)

	stops, err := engine.StopTokenIDs()
	if err != nil {
		fmt.Fprintf(os.Stderr, "stop tokens: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Engine.StopTokenIDs:  %v\n", stops)
}
