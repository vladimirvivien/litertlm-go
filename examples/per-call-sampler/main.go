// per-call-sampler runs the same prompt three times against one
// Client, each Generate call overriding the sampler shape via
// WithSampler. The output divergence between the three replies is
// the demonstration.
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
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	prompt := flag.String("prompt", "Write a haiku about the changing seasons.", "prompt issued for all three sampler shapes")
	seed := flag.Int("seed", 1, "sampler seed; affects the TopP cases only")
	maxTokens := flag.Int("max-tokens", 96, "per-call decode cap (WithMaxOutputTokens)")
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

	shapes := []struct {
		label   string
		sampler litertlm.SamplerParams
	}{
		{
			label: "Deterministic (TopP=0.95, Temp=0)",
			sampler: litertlm.SamplerParams{
				Type:        litertlm.SamplerTopP,
				TopK:        40,
				TopP:        0.95,
				Temperature: 0,
				Seed:        int32(*seed),
			},
		},
		{
			label: "Balanced (TopP=0.9, Temp=0.7)",
			sampler: litertlm.SamplerParams{
				Type:        litertlm.SamplerTopP,
				TopK:        40,
				TopP:        0.9,
				Temperature: 0.7,
				Seed:        int32(*seed),
			},
		},
		{
			label: "Creative (TopP=0.95, Temp=1.2)",
			sampler: litertlm.SamplerParams{
				Type:        litertlm.SamplerTopP,
				TopK:        40,
				TopP:        0.95,
				Temperature: 1.2,
				Seed:        int32(*seed),
			},
		},
	}

	fmt.Printf("prompt: %s\n\n", *prompt)

	for i, s := range shapes {
		fmt.Printf("=== Run %d: %s ===\n", i+1, s.label)
		text, err := client.Generate(ctx, *prompt,
			litertlm.WithSampler(s.sampler),
			litertlm.WithMaxOutputTokens(*maxTokens),
		)
		if err != nil {
			fmt.Fprintf(os.Stderr, "generate %d: %v\n", i+1, err)
			os.Exit(1)
		}
		fmt.Println(text)
		fmt.Println()
	}
}
