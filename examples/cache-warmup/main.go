// cache-warmup contrasts a cold and warm WithCacheDir load by
// running litertlm.New + a one-shot Generate twice against the same
// cache directory and reporting the wall-clock delta.
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
	cacheDir := flag.String("cache-dir", "", "directory passed to WithCacheDir; empty creates a fresh temp dir removed at exit")
	keepCache := flag.Bool("keep-cache", false, "skip cleanup of an auto-created temp cache dir")
	prompt := flag.String("prompt", "The capital of France is", "completion-style prompt used for the one-shot Generate after each load")
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

	ctx := context.Background()

	fmt.Println("\n=== Run 1 (cold) ===")
	cold, err := runOnce(ctx, *libPath, *model, *backend, dir, *prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "run 1: %v\n", err)
		os.Exit(1)
	}
	printTimings(cold)

	fmt.Println("\n=== Run 2 (warm) ===")
	warm, err := runOnce(ctx, *libPath, *model, *backend, dir, *prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "run 2: %v\n", err)
		os.Exit(1)
	}
	printTimings(warm)

	fmt.Println("\n=== Delta (cold - warm) ===")
	fmt.Printf("  litertlm.New: %v\n", cold.newTime-warm.newTime)
	fmt.Printf("  Generate:     %v\n", cold.genTime-warm.genTime)

	listArtefacts(dir)
}

type timings struct {
	newTime time.Duration
	genTime time.Duration
	bench   *litertlm.Benchmark
}

func runOnce(ctx context.Context, libPath, modelPath, backend, cacheDir, prompt string) (*timings, error) {
	t0 := time.Now()
	client, err := litertlm.New(ctx,
		litertlm.WithLib(libPath),
		litertlm.WithModel(modelPath),
		litertlm.WithBackend(backend),
		litertlm.WithCacheDir(cacheDir),
		litertlm.WithBenchmarkEnabled(),
	)
	if err != nil {
		return nil, fmt.Errorf("new client: %w", err)
	}
	newTime := time.Since(t0)
	defer client.Close()

	t1 := time.Now()
	resp, err := client.GenerateResponse(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("generate: %w", err)
	}
	genTime := time.Since(t1)

	return &timings{newTime: newTime, genTime: genTime, bench: resp.Benchmark()}, nil
}

func printTimings(t *timings) {
	fmt.Printf("  litertlm.New: %v\n", t.newTime)
	fmt.Printf("  Generate:     %v\n", t.genTime)
	if t.bench == nil {
		return
	}
	fmt.Printf("  TimeToFirstToken: %v\n", t.bench.TimeToFirstToken)
	if len(t.bench.PrefillTokensPerSec) > 0 {
		fmt.Printf("  Prefill: %.1f tok/s (%d tokens)\n",
			t.bench.PrefillTokensPerSec[0], t.bench.PrefillTokenCounts[0])
	}
	if len(t.bench.DecodeTokensPerSec) > 0 {
		fmt.Printf("  Decode:  %.1f tok/s (%d tokens)\n",
			t.bench.DecodeTokensPerSec[0], t.bench.DecodeTokenCounts[0])
	}
}

func resolveCacheDir(flagDir string, keep bool) (string, func(), error) {
	if flagDir != "" {
		if err := os.MkdirAll(flagDir, 0o755); err != nil {
			return "", func() {}, err
		}
		return flagDir, func() {}, nil
	}
	d, err := os.MkdirTemp("", "litertlm-cache-warmup-*")
	if err != nil {
		return "", func() {}, err
	}
	if keep {
		return d, func() { fmt.Printf("temp cache dir kept at %s\n", d) }, nil
	}
	return d, func() { _ = os.RemoveAll(d) }, nil
}

func listArtefacts(dir string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	if len(entries) == 0 {
		fmt.Printf("\nno artefacts written to %s — the engine did not use the cache dir on this backend.\n", dir)
		return
	}
	fmt.Printf("\nartefacts written to %s:\n", dir)
	for _, e := range entries {
		info, err := e.Info()
		if err != nil {
			continue
		}
		fmt.Printf("  %s (%d bytes)\n", e.Name(), info.Size())
	}
}
