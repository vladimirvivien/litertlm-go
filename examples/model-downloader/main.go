// model-downloader demonstrates automated downloading and execution of .litertlm models
// using litertlm.FetchModel with terminal progress reporting.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	modelID := flag.String("model", "litert-community/gemma3-1b-it-int4", "Hugging Face model ID or direct HTTP(S) URL")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.16.0)")
	destDir := flag.String("dir", "", "local directory to cache downloaded models (optional)")
	token := flag.String("token", "", "Hugging Face auth token for gated models (optional, falls back to HF_TOKEN env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "path to LiteRT-LM shared library directory")
	prompt := flag.String("prompt", "What is the speed of light?", "prompt to run after download")
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

	resolvedModel := *modelID
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

	// 1. Download model with progress callback
	fmt.Printf("Fetching model %q...\n", *modelID)
	var opts []litertlm.ModelFetchOption
	if *destDir != "" {
		opts = append(opts, litertlm.WithModelDir(*destDir))
	}
	if *token != "" {
		opts = append(opts, litertlm.WithModelAuthToken(*token))
	}

	opts = append(opts, litertlm.WithModelProgress(func(downloaded, total int64, pct float64) {
		if total > 0 {
			mbDown := float64(downloaded) / (1024 * 1024)
			mbTotal := float64(total) / (1024 * 1024)
			bars := min(int(pct/2.5), 40)
			barStr := strings.Repeat("=", bars) + strings.Repeat(" ", 40-bars)
			fmt.Printf("\r[%s] %.1f%% (%.1f / %.1f MB)", barStr, pct, mbDown, mbTotal)
		} else {
			mbDown := float64(downloaded) / (1024 * 1024)
			fmt.Printf("\rDownloaded %.1f MB", mbDown)
		}
	}))

	modelPath, fetchErr := litertlm.FetchModel(ctx, *modelID, opts...)
	if fetchErr != nil {
		fmt.Println()
		log.Fatalf("Model download failed: %v", fetchErr)
	}
	fmt.Printf("\nModel ready at: %s\n\n", modelPath)

	// 2. Automatically fetch C-API libraries if LITERTLM_LIB is not provided
	if *libPath == "" {
		stagedLib, errLib := litertlm.FetchLib("", "", "v0.16.0")
		if errLib == nil {
			*libPath = stagedLib
		}
	}

	// 3. Initialize client and generate
	client, err := litertlm.New(ctx,
		litertlm.WithLib(resolvedLib),
		litertlm.WithModel(modelPath),
		litertlm.WithBackend(*backend),
	)
	if err != nil {
		log.Fatalf("Client initialization failed: %v", err)
	}
	defer client.Close()

	fmt.Printf("Prompt: %s\n\nResponse:\n", *prompt)
	text, err := client.Generate(ctx, *prompt)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}
	fmt.Println(text)
}
