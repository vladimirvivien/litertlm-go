// speculative compares decode throughput with and without
// WithSpeculativeDecodingEnabled. Loads the model twice (the
// speculative flag is engine-level, not per-call), runs the same
// prompt against each Client, and prints a side-by-side benchmark
// summary plus the computed speedup ratio.
//
// Requires a model that supports speculative (multi-token-prediction)
// decoding, e.g. Gemma 4.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

type runResult struct {
	wallTime time.Duration
	bench    *litertlm.Benchmark
	text     string
}

func main() {
	model := flag.String("model", "", "path to a .litertlm model that supports speculative decoding (e.g. Gemma 4)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	prompt := flag.String("prompt", "The capital of France is", "prompt text (completion-style works best with chat-tuned models on the raw Generate path)")
	maxOutput := flag.Int("max-output", 256, "max output tokens per call")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "--model is required")
		os.Exit(2)
	}

	ctx := context.Background()

	fmt.Println("=== Without speculative decoding ===")
	off, err := runOnce(ctx, *libPath, *model, *backend, *prompt, *maxOutput, false)
	if err != nil {
		fmt.Fprintf(os.Stderr, "baseline run: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(off.text)
	printBenchmark(off, "")

	fmt.Println("\n=== With speculative decoding (multi-token prediction) ===")
	on, err := runOnce(ctx, *libPath, *model, *backend, *prompt, *maxOutput, true)
	if err != nil {
		fmt.Fprintf(os.Stderr, "speculative run: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(on.text)
	printBenchmark(on, "")

	fmt.Println("\n=== Speedup ===")
	printSpeedup(off, on)
}

func runOnce(ctx context.Context, libPath, model, backend, prompt string, maxOutput int, speculative bool) (runResult, error) {
	opts := []litertlm.Option{
		litertlm.WithLib(libPath),
		litertlm.WithModel(model),
		litertlm.WithBackend(backend),
		litertlm.WithBenchmarkEnabled(),
	}
	if speculative {
		opts = append(opts, litertlm.WithSpeculativeDecodingEnabled(true))
	}

	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	client, err := litertlm.New(ctx, opts...)
	if err != nil {
		return runResult{}, fmt.Errorf("new client: %w", err)
	}
	defer client.Close()

	start := time.Now()
	resp, err := client.GenerateResponse(ctx, prompt, litertlm.WithMaxOutputTokens(maxOutput))
	wall := time.Since(start)
	if err != nil {
		return runResult{}, fmt.Errorf("generate: %w", err)
	}
	bench := resp.Benchmark()
	if bench == nil {
		return runResult{}, fmt.Errorf("no benchmark captured (WithBenchmarkEnabled missing?)")
	}
	return runResult{wallTime: wall, bench: bench, text: resp.Text()}, nil
}

func printBenchmark(r runResult, indent string) {
	b := r.bench
	fmt.Printf("%s  Init time:               %v\n", indent, b.TotalInitTime)
	fmt.Printf("%s  Time to first token:     %v\n", indent, b.TimeToFirstToken)
	fmt.Printf("%s  Wall-clock generate:     %v\n", indent, r.wallTime)
	for i, tps := range b.PrefillTokensPerSec {
		fmt.Printf("%s  Prefill turn %d:          %.1f tok/s  (%d tokens)\n",
			indent, i, tps, b.PrefillTokenCounts[i])
	}
	for i, tps := range b.DecodeTokensPerSec {
		fmt.Printf("%s  Decode  turn %d:          %.1f tok/s  (%d tokens)\n",
			indent, i, tps, b.DecodeTokenCounts[i])
	}
}

func printSpeedup(off, on runResult) {
	offDecode := totalDecodeTokens(off.bench)
	onDecode := totalDecodeTokens(on.bench)

	offTPS := avg(off.bench.DecodeTokensPerSec)
	onTPS := avg(on.bench.DecodeTokensPerSec)

	fmt.Printf("  Decode tokens (off / on):  %d / %d\n", offDecode, onDecode)
	fmt.Printf("  Decode tok/s  (off / on):  %.1f / %.1f", offTPS, onTPS)
	if offTPS > 0 {
		fmt.Printf("   (%.2f×)\n", onTPS/offTPS)
	} else {
		fmt.Println()
	}
	if off.wallTime > 0 {
		fmt.Printf("  Wall-clock    (off / on):  %v / %v   (%.2f×)\n",
			off.wallTime, on.wallTime,
			float64(off.wallTime)/float64(on.wallTime))
	}
}

func totalDecodeTokens(b *litertlm.Benchmark) int {
	var n int
	for _, c := range b.DecodeTokenCounts {
		n += c
	}
	return n
}

func avg(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	var sum float64
	for _, x := range xs {
		sum += x
	}
	return sum / float64(len(xs))
}
