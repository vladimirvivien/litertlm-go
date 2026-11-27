// metal-residency demonstrates initializing Metal residency sets at startup
// via WithMetalResidencySet and dynamically toggling residency behavior at
// runtime via Client.SetMetalResidency.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	model := flag.String("model", "", "path to a .litertlm model file (or set LITERTLM_MODEL)")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.19.1)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "gpu", "inference backend (gpu | cpu)")
	prompt := flag.String("prompt", "In one sentence, explain why keeping GPU weights resident improves decode latency.", "prompt to run before toggling residency")
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
		fmt.Fprintln(os.Stderr, "-model or -get-model (or LITERTLM_MODEL env) is required")
		os.Exit(2)
	}

	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	client, err := litertlm.New(ctx,
		litertlm.WithLib(resolvedLib),
		litertlm.WithModel(resolvedModel),
		litertlm.WithBackend(*backend),
		litertlm.WithMetalResidencySet(true),
		litertlm.WithDefaultSampler(litertlm.SamplerParams{
			Type:        litertlm.SamplerTopP,
			TopK:        40,
			TopP:        0.95,
			Temperature: 0,
			Seed:        1,
		}),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = client.Close() }()

	text, err := client.Generate(ctx, *prompt, litertlm.WithMaxOutputTokens(64))
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Initial reply (Metal residency=true):\n%s\n\n", text)

	if err := client.SetMetalResidency(false); err != nil {
		fmt.Fprintf(os.Stderr, "SetMetalResidency(false): %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Updated runtime Metal residency -> false (active on macOS/iOS Metal GPU; ignored safely elsewhere).")

	if err := client.SetMetalResidency(true); err != nil {
		fmt.Fprintf(os.Stderr, "SetMetalResidency(true): %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Restored runtime Metal residency -> true.")
}
