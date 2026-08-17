// gpu demonstrates GPU-accelerated local inference: load the
// GPU-capable LiteRT-LM build, configure an engine for the GPU
// backend, and stream tokens from a single prompt.
//
// See README.md in this directory for the GPU-specific build and the
// extra runtime libraries that must be staged in LITERTLM_LIB.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	model := flag.String("model", "", "path to .litertlm model file (falls back to LITERTLM_MODEL env)")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	prompt := flag.String("prompt", "Summarise Go's approach to concurrency in one paragraph.", "prompt text")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the GPU-capable LiteRT-LM shared library + GPU plugin .so/.dylib/.dll files (falls back to LITERTLM_LIB env)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.16.0)")
	maxTokens := flag.Int("max", 1024, "max total tokens (prompt + output); must be >= the model's smallest prefill signature, typically 128")
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

	if err := litertlm.Load(resolvedLib, "gpu", ""); err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer litertlm.Close()

	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	settings, err := litertlm.NewEngineSettings(resolvedModel, "gpu", nil, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "settings: %v\n", err)
		os.Exit(1)
	}
	defer settings.Delete()
	settings.SetMaxNumTokens(*maxTokens)

	engine, err := litertlm.NewEngine(settings)
	if err != nil {
		fmt.Fprintf(os.Stderr, "engine: %v\n", err)
		os.Exit(1)
	}
	defer engine.Delete()

	session, err := engine.NewSession(0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "session: %v\n", err)
		os.Exit(1)
	}
	defer session.Delete()

	in, err := litertlm.NewTextInputString(*prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "input: %v\n", err)
		os.Exit(1)
	}
	defer in.Delete()

	inputs := []litertlm.InputData{in}
	for chunk := range session.GenerateContentStreamCh(inputs) {
		if chunk.Err != nil {
			fmt.Fprintf(os.Stderr, "\nstream error: %v\n", chunk.Err)
			os.Exit(1)
		}
		fmt.Print(chunk.Text)
		if chunk.Final {
			fmt.Println()
		}
	}
}
