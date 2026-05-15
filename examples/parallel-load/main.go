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
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	parallel := flag.Bool("parallel", true, "parallel section loading mode (default true matches engine default)")
	prompt := flag.String("prompt", "The capital of France is", "short prompt to confirm the engine loaded correctly")
	cacheDir := flag.String("cache-dir", "", "directory passed to WithCacheDir; empty creates a fresh temp dir removed at exit")
	keepCache := flag.Bool("keep-cache", false, "skip cleanup of an auto-created temp cache dir")
	maxTokens := flag.Int("max-tokens", 4096, "engine total token budget")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "--model is required")
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
		litertlm.WithLib(*libPath),
		litertlm.WithModel(*model),
		litertlm.WithBackend(*backend),
		litertlm.WithCacheDir(dir),
		litertlm.WithMaxTokens(*maxTokens),
		litertlm.WithBenchmarkEnabled(),
	}
	if !*parallel {
		opts = append(opts, litertlm.WithParallelSectionLoading(false))
	}

	ctx := context.Background()

	t0 := time.Now()
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
