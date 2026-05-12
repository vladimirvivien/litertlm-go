// structured demonstrates type-safe structured-output extraction with
// litertlm.GenerateData[T]. The model is asked for a Recipe; the
// helper injects a JSON-shape instruction, generates, extracts the
// JSON from the (possibly markdown-fenced, prose-padded) reply, and
// unmarshals into the typed Go struct.
//
// Local LLMs occasionally produce invalid JSON; bump `-retries` if
// the first attempt fails.
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
	model := flag.String("model", "", "path to .litertlm model file")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries (falls back to LITERTLM_LIB env)")
	prompt := flag.String("prompt", "Recipe for chocolate chip cookies.", "what to extract")
	retries := flag.Int("retries", 2, "max retry attempts on parse failure")
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
