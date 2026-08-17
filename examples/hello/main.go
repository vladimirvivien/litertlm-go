// hello demonstrates a minimal synchronous inference using the
// high-level Client API. For the equivalent low-level (Session-based)
// flow, see examples/prefill-decode.
//
// See README.md in this directory for prerequisites and usage.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	model := flag.String("model", "", "path to .litertlm model file (falls back to LITERTLM_MODEL env)")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	prompt := flag.String("prompt", "The capital of France is", "prompt text")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries (falls back to LITERTLM_LIB env)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.16.0)")
	maxTokens := flag.Int("max", 1024, "max total tokens (prompt + output); must be >= the model's smallest prefill signature, typically 128")
	logLevel := flag.String("loglevel", "quiet", "LiteRT-LM log severity floor: verbose | debug | info | warning | error | fatal | quiet (also accepts the numeric form)")
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

	level, err := parseLogLevel(*logLevel)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	litertlm.SetMinLogLevel(level)

	client, err := litertlm.New(ctx,
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

	text, err := client.Generate(ctx, *prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(text)
}

func parseLogLevel(s string) (litertlm.LogLevel, error) {
	switch strings.ToLower(s) {
	case "verbose", "0":
		return litertlm.LogVerbose, nil
	case "debug", "1":
		return litertlm.LogDebug, nil
	case "info", "2":
		return litertlm.LogInfo, nil
	case "warning", "warn", "3":
		return litertlm.LogWarning, nil
	case "error", "4":
		return litertlm.LogError, nil
	case "fatal", "5":
		return litertlm.LogFatal, nil
	case "quiet", "1000":
		return litertlm.LogQuiet, nil
	}
	return 0, fmt.Errorf("unknown log level %q (use verbose | debug | info | warning | error | fatal | quiet)", s)
}
