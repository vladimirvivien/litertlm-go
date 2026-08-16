// logging demonstrates the LiteRT-LM log severity floor. The level
// is set before New via SetMinLogLevel, then lowered between
// subsequent Generate calls to show that the floor is a
// process-global setting that can change at any time.
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
	model := flag.String("model", "", "path to .litertlm model file (required)")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.16.0)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	startLevel := flag.String("loglevel", "info", "starting log level: verbose | debug | info | warning | error | fatal | quiet (also accepts the numeric form)")
	prompt := flag.String("prompt", "The capital of France is", "short prompt issued for each Generate call")
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

	start, err := parseLogLevel(*startLevel)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	litertlm.SetMinLogLevel(start)

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

	runAt(ctx, client, *prompt, 1, start)

	litertlm.SetMinLogLevel(litertlm.LogError)
	runAt(ctx, client, *prompt, 2, litertlm.LogError)

	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	runAt(ctx, client, *prompt, 3, litertlm.LogQuiet)
}

func runAt(ctx context.Context, client *litertlm.Client, prompt string, n int, level litertlm.LogLevel) {
	fmt.Printf("\n=== Generate #%d: log level = %s ===\n", n, level)
	resp, err := client.GenerateResponse(ctx, prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate #%d: %v\n", n, err)
		os.Exit(1)
	}
	fmt.Printf("reply: %s\n", resp.Text())
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
