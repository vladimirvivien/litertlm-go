// parallel-load measures litertlm.New wall-clock under the engine's
// default parallel section loading vs the serial path forced by
// WithParallelSectionLoading(false). One mode per invocation: invoke
// twice with different -parallel values and compare. The example uses
// a fresh cache dir each invocation so the compile-cache build cost
// is paid in both runs.
//
// See README.md in this directory for prerequisites and usage.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	model := flag.String("model", "", "path to .litertlm model file (required)")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.16.0)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	parallel := flag.Bool("parallel", true, "parallel section loading mode (default true matches engine default)")
	prompt := flag.String("prompt", "The capital of France is", "short prompt to confirm the engine loaded correctly")
	cacheDir := flag.String("cache-dir", "", "directory passed to WithCacheDir; empty creates a fresh temp dir removed at exit")
	keepCache := flag.Bool("keep-cache", false, "skip cleanup of an auto-created temp cache dir")
	maxTokens := flag.Int("max-tokens", 4096, "engine total token budget")
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

	dir, cleanup, err := resolveCacheDir(*cacheDir, *keepCache)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cache dir: %v\n", err)
		os.Exit(1)
	}
	defer cleanup()
	fmt.Printf("cache dir: %s\n", dir)
	fmt.Printf("parallel:  %v\n", *parallel)

	opts := []litertlm.Option{
		litertlm.WithLib(resolvedLib),
		litertlm.WithModel(resolvedModel),
		litertlm.WithBackend(*backend),
		litertlm.WithCacheDir(dir),
		litertlm.WithMaxTokens(*maxTokens),
		litertlm.WithBenchmarkEnabled(),
	}
	if !*parallel {
		opts = append(opts, litertlm.WithParallelSectionLoading(false))
	}

	t0 := time.Now()
	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	client, err := litertlm.New(ctx, opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	newTime := time.Since(t0)
	defer client.Close()

	t1 := time.Now()
	resp, err := client.GenerateResponse(ctx, *prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}
	genTime := time.Since(t1)

	fmt.Println("\n=== Results ===")
	fmt.Printf("  litertlm.New: %v\n", newTime)
	fmt.Printf("  Generate:     %v\n", genTime)
	if b := resp.Benchmark(); b != nil {
		fmt.Printf("  TimeToFirstToken: %v\n", b.TimeToFirstToken)
		if len(b.PrefillTokensPerSec) > 0 {
			fmt.Printf("  Prefill: %.1f tok/s (%d tokens)\n",
				b.PrefillTokensPerSec[0], b.PrefillTokenCounts[0])
		}
		if len(b.DecodeTokensPerSec) > 0 {
			fmt.Printf("  Decode:  %.1f tok/s (%d tokens)\n",
				b.DecodeTokensPerSec[0], b.DecodeTokenCounts[0])
		}
	}
}

func resolveCacheDir(flagDir string, keep bool) (string, func(), error) {
	if flagDir != "" {
		if err := os.MkdirAll(flagDir, 0o755); err != nil {
			return "", func() {}, err
		}
		return flagDir, func() {}, nil
	}
	d, err := os.MkdirTemp("", "litertlm-parallel-load-*")
	if err != nil {
		return "", func() {}, err
	}
	if keep {
		return d, func() { fmt.Printf("temp cache dir kept at %s\n", d) }, nil
	}
	return d, func() { _ = os.RemoveAll(d) }, nil
}
