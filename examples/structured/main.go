// structured demonstrates type-safe structured-output extraction with
// litertlm.GenerateData[T]. The model is asked for a Recipe; the
// helper returns a typed *Recipe populated from the model's response.
//
// GenerateData[T] routes through a synthesized tool-call capture when
// T is a struct, with a prompt-engineered fallback for other cases or
// when the model declines to call the tool. See docs/structured-output.md
// for the full pipeline; -retries N controls fallback-path attempts.
//
// See README.md in this directory for prerequisites and usage.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

// Recipe is the target type for GenerateData[T]. JSON tags drive
// both the injected shape hint and the unmarshal step; default
// lowercased Go names work too.
type Recipe struct {
	Title       string   `json:"title"`
	Ingredients []string `json:"ingredients"`
	Steps       []string `json:"steps"`
}

func main() {
	model := flag.String("model", "", "path to .litertlm model file (falls back to LITERTLM_MODEL env)")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries (falls back to LITERTLM_LIB env)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.16.0)")
	prompt := flag.String("prompt", "Recipe for chocolate chip cookies.", "what to extract")
	retries := flag.Int("retries", 2, "max retry attempts on parse failure")
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
		litertlm.WithMaxTokens(2048),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	recipe, err := litertlm.GenerateData[Recipe](ctx, client, *prompt,
		litertlm.WithRetries(*retries),
	)
	if err != nil {
		// Distinguish parse failures (model produced text but it
		// couldn't be unmarshalled) from generate failures (call
		// itself errored).
		var gd *litertlm.GenerateDataError
		if errors.As(err, &gd) && gd.Phase == "parse" {
			fmt.Fprintf(os.Stderr, "parse failed after %d attempt(s); raw model output:\n%s\n",
				gd.Attempts, gd.Raw)
		} else {
			fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		}
		os.Exit(1)
	}

	pretty, _ := json.MarshalIndent(recipe, "", "  ")
	fmt.Println(string(pretty))
}
