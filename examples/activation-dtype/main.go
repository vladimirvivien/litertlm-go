// activation-dtype contrasts the default activation precision with a
// caller-selected WithActivationDataType. Both runs issue the same
// Generate against the same model so prefill / decode tokens-per-sec
// are directly comparable.
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
	dtype := flag.Int("dtype", 1, "activation dtype for run 2: 0=F32, 1=F16, 2=I16, 3=I8")
	prompt := flag.String("prompt", "Explain in three sentences what makes Go a popular systems programming language.", "completion-style prompt")
	cacheDir := flag.String("cache-dir", "", "directory passed to WithCacheDir; empty creates a fresh temp dir removed at exit")
	keepCache := flag.Bool("keep-cache", false, "skip cleanup of an auto-created temp cache dir")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "--model is required")
		os.Exit(2)
	}
	if *dtype < 0 || *dtype > 3 {
		fmt.Fprintln(os.Stderr, "--dtype must be 0, 1, 2, or 3")
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

	fmt.Println("\n=== Run 1: default activation dtype ===")
	def, err := runOnce(ctx, *libPath, *model, *backend, dir, nil, *prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "run 1: %v\n", err)
		os.Exit(1)
	}
	printRun(def)

	fmt.Printf("\n=== Run 2: WithActivationDataType(%d) — %s ===\n", *dtype, dtypeName(*dtype))
	sel, err := runOnce(ctx, *libPath, *model, *backend, dir, dtype, *prompt)
	if err != nil {
		fmt.Printf("engine construction failed: %v\n", err)
		fmt.Printf("dtype %d (%s) is not supported on backend %q.\n", *dtype, dtypeName(*dtype), *backend)
		return
	}
	printRun(sel)

	fmt.Println("\n=== Delta (default - selected) ===")
	fmt.Printf("  TimeToFirstToken: %v\n", def.bench.TimeToFirstToken-sel.bench.TimeToFirstToken)
	if len(def.bench.PrefillTokensPerSec) > 0 && len(sel.bench.PrefillTokensPerSec) > 0 {
		fmt.Printf("  Prefill tok/s:    %+.1f\n",
			sel.bench.PrefillTokensPerSec[0]-def.bench.PrefillTokensPerSec[0])
	}
	if len(def.bench.DecodeTokensPerSec) > 0 && len(sel.bench.DecodeTokensPerSec) > 0 {
		fmt.Printf("  Decode  tok/s:    %+.1f\n",
			sel.bench.DecodeTokensPerSec[0]-def.bench.DecodeTokensPerSec[0])
	}
}

func resolveCacheDir(flagDir string, keep bool) (string, func(), error) {
	if flagDir != "" {
		if err := os.MkdirAll(flagDir, 0o755); err != nil {
			return "", func() {}, err
		}
		return flagDir, func() {}, nil
	}
	d, err := os.MkdirTemp("", "litertlm-activation-dtype-*")
	if err != nil {
		return "", func() {}, err
	}
	if keep {
		return d, func() { fmt.Printf("temp cache dir kept at %s\n", d) }, nil
	}
	return d, func() { _ = os.RemoveAll(d) }, nil
}

type runResult struct {
	newTime time.Duration
	genTime time.Duration
	bench   *litertlm.Benchmark
}

func runOnce(ctx context.Context, libPath, modelPath, backend, cacheDir string, dtype *int, prompt string) (*runResult, error) {
	opts := []litertlm.Option{
		litertlm.WithLib(libPath),
		litertlm.WithModel(modelPath),
		litertlm.WithBackend(backend),
		litertlm.WithCacheDir(cacheDir),
		litertlm.WithBenchmarkEnabled(),
	}
	if dtype != nil {
		opts = append(opts, litertlm.WithActivationDataType(*dtype))
	}

	t0 := time.Now()
	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	client, err := litertlm.New(ctx, opts...)
	if err != nil {
		return nil, err
	}
	newTime := time.Since(t0)
	defer client.Close()

	t1 := time.Now()
	resp, err := client.GenerateResponse(ctx, prompt)
	if err != nil {
		return nil, err
	}
	genTime := time.Since(t1)
	return &runResult{newTime: newTime, genTime: genTime, bench: resp.Benchmark()}, nil
}

func printRun(r *runResult) {
	fmt.Printf("  litertlm.New: %v\n", r.newTime)
	fmt.Printf("  Generate:     %v\n", r.genTime)
	if r.bench != nil {
		fmt.Printf("  TimeToFirstToken: %v\n", r.bench.TimeToFirstToken)
		if len(r.bench.PrefillTokensPerSec) > 0 {
			fmt.Printf("  Prefill: %.1f tok/s (%d tokens)\n",
				r.bench.PrefillTokensPerSec[0], r.bench.PrefillTokenCounts[0])
		}
		if len(r.bench.DecodeTokensPerSec) > 0 {
			fmt.Printf("  Decode:  %.1f tok/s (%d tokens)\n",
				r.bench.DecodeTokensPerSec[0], r.bench.DecodeTokenCounts[0])
		}
	}
}

func dtypeName(t int) string {
	switch t {
	case 0:
		return "F32"
	case 1:
		return "F16"
	case 2:
		return "I16"
	case 3:
		return "I8"
	default:
		return "unknown"
	}
}
