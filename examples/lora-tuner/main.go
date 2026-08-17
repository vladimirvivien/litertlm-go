// lora-tuner demonstrates how to configure LoRA ranks and parameters
// on EngineSettings at engine initialization.
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
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries")
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

	// Load C-API libraries
	if err := litertlm.Load(resolvedLib, *backend, ""); err != nil {
		fmt.Fprintf(os.Stderr, "failed to load libraries: %v\n", err)
		os.Exit(1)
	}

	// 1. Create EngineSettings low-level config
	fmt.Println("Creating EngineSettings...")
	settings, err := litertlm.NewEngineSettings(resolvedModel, *backend, nil, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "NewEngineSettings: %v\n", err)
		os.Exit(1)
	}
	defer settings.Delete()

	// 2. Configure LoRA ranks
	// Set the current base rank
	fmt.Println("Configuring LoRA Rank to 8...")
	settings.SetLoraRank(8)

	// Declare all supported adapter ranks (e.g. models can load adapters with rank 8 or 16)
	fmt.Println("Declaring supported ranks [8, 16]...")
	if loraErr := settings.SetSupportedLoraRanks([]int{8, 16}); loraErr != nil {
		fmt.Fprintf(os.Stderr, "SetSupportedLoraRanks: %v\n", loraErr)
		// Note: Will log warnings or return error codes on CPU/unsupported models
	}

	// 3. Compile the Engine
	fmt.Println("Compiling the engine with LoRA settings...")
	engine, err := litertlm.NewEngine(settings)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to compile engine: %v\n", err)
		os.Exit(1)
	}
	defer engine.Delete()

	fmt.Println("Engine compiled successfully with LoRA configurations.")
}
