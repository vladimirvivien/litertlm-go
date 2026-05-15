// benchmarks demonstrates per-generation benchmark collection on the
// high-level Client API. Pair WithBenchmarkEnabled at New time with
// Response.Benchmark() on each call to read pure-Go metrics.
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
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries (falls back to LITERTLM_LIB env)")
	prompt := flag.String("prompt", "The capital of France is", "prompt text (use a completion-style prompt for chat-tuned models routed through raw Generate)")
	turns := flag.Int("turns", 2, "number of generations to run before printing aggregate metrics")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "--model is required")
		os.Exit(2)
	}

	ctx := context.Background()
	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	client, err := litertlm.New(ctx,
		litertlm.WithLib(*libPath),
		litertlm.WithModel(*model),
		litertlm.WithBackend(*backend),
		litertlm.WithBenchmarkEnabled(),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	for i := 1; i <= *turns; i++ {
		fmt.Printf("=== Turn %d ===\n", i)
		resp, err := client.GenerateResponse(ctx, *prompt)
		if err != nil {
			fmt.Fprintf(os.Stderr, "generate: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(resp.Text())

		b := resp.Benchmark()
		if b == nil {
			fmt.Fprintln(os.Stderr, "no benchmark captured (was WithBenchmarkEnabled set?)")
			os.Exit(1)
		}
		printBenchmark(b)
	}
}

func printBenchmark(b *litertlm.Benchmark) {
	fmt.Println("--- Benchmark ---")
	fmt.Printf("  Total init time:       %v\n", b.TotalInitTime)
	fmt.Printf("  Time to first token:   %v\n", b.TimeToFirstToken)
	fmt.Printf("  Prefill turns: %d   Decode turns: %d\n", b.PrefillTurns, b.DecodeTurns)
	for i, tps := range b.PrefillTokensPerSec {
		fmt.Printf("  Prefill turn %d:  %.1f tok/s  (%d tokens)\n",
			i, tps, b.PrefillTokenCounts[i])
	}
	for i, tps := range b.DecodeTokensPerSec {
		fmt.Printf("  Decode  turn %d:  %.1f tok/s  (%d tokens)\n",
			i, tps, b.DecodeTokenCounts[i])
	}
	fmt.Println()
}
