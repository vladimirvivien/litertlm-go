// constrained-json demonstrates grammar-enforced generation using
// WithConstrainedDecoding, WithConstraintProvider(ConstraintProviderLLGuidance),
// ConstraintJSONSchema, and ConstraintRegex.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

const personSchema = `{
  "type": "object",
  "properties": {
    "name": { "type": "string" },
    "age": { "type": "integer" }
  },
  "required": ["name", "age"],
  "additionalProperties": false
}`

func main() {
	model := flag.String("model", "", "path to .litertlm model file (falls back to LITERTLM_MODEL env)")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.19.1)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
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
		litertlm.WithMaxTokens(1024),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = client.Close() }()

	chat, err := client.NewChat(ctx,
		litertlm.WithConstrainedDecoding(true),
		litertlm.WithConstraintProvider(litertlm.ConstraintProviderLLGuidance),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new chat: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = chat.Close() }()

	reply1, err := chat.Send(ctx,
		"Extract the person's name and age from: Ada Lovelace was 36 years old.",
		litertlm.WithMaxOutputTokens(64),
		litertlm.WithConstraint(litertlm.ConstraintJSONSchema, personSchema),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "json schema turn: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("JSON Schema output: %s\n", reply1.Text())

	reply2, err := chat.Send(ctx,
		"Is Paris in France? Answer yes or no.",
		litertlm.WithMaxOutputTokens(16),
		litertlm.WithConstraint(litertlm.ConstraintRegex, `(yes|no)`),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "regex turn: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Regex output: %s\n", reply2.Text())
}
