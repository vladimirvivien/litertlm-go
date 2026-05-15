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
	model := flag.String("model", "", "path to .litertlm model file")
	prompt := flag.String("prompt", "The capital of France is", "prompt text")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries (falls back to LITERTLM_LIB env)")
	maxTokens := flag.Int("max", 1024, "max total tokens (prompt + output); must be >= the model's smallest prefill signature, typically 128")
	logLevel := flag.String("loglevel", "quiet", "LiteRT-LM log severity floor: verbose | debug | info | warning | error | fatal | quiet (also accepts the numeric form)")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "--model is required")
		os.Exit(2)
	}
	level, err := parseLogLevel(*logLevel)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	litertlm.SetMinLogLevel(level)

	ctx := context.Background()
	client, err := litertlm.New(ctx,
		litertlm.WithLib(*libPath),
		litertlm.WithModel(*model),
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
