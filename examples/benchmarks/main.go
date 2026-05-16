// benchmarks contrasts two per-generation benchmark surfaces on the
// same Client and Engine:
//
//   - The high-level path: Response.Benchmark() snapshots into a
//     pure-Go *Benchmark when Client was built with
//     WithBenchmarkEnabled.
//   - The low-level path: Session.BenchmarkInfo() returns a C-side
//     handle the caller deletes.
//
// See README.md in this directory for prerequisites and usage.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	model := flag.String("model", "", "path to .litertlm model file (required)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	prompt := flag.String("prompt", "The capital of France is", "completion-style prompt; Generate does not apply the chat template")
	turns := flag.Int("turns", 2, "number of high-level Generate calls before the low-level section")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "--model is required")
		os.Exit(2)
	}

	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	ctx := context.Background()
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
		fmt.Printf("=== High-level turn %d (Response.Benchmark) ===\n", i)
		var resp *litertlm.Response
		resp, err = client.GenerateResponse(ctx, *prompt)
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
		printGoBenchmark(b)
	}

	fmt.Println("=== Low-level (Session.BenchmarkInfo) ===")
	sess, err := client.Engine().NewSession(0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new session: %v\n", err)
		os.Exit(1)
	}
	defer sess.Delete()

	resp, err := sess.GenerateContent([]litertlm.InputData{
		litertlm.NewTextInputString(*prompt),
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "session generate: %v\n", err)
		os.Exit(1)
	}
	defer resp.Delete()
	fmt.Println(resp.Text(0))

	bi, err := sess.BenchmarkInfo()
	if err != nil {
		fmt.Fprintf(os.Stderr, "benchmark info: %v\n", err)
		os.Exit(1)
	}
	defer bi.Delete()
	printCBenchmarkInfo(bi)
}

func printGoBenchmark(b *litertlm.Benchmark) {
	fmt.Println("--- *Benchmark (Go) ---")
	fmt.Printf("  Total init time:     %v\n", b.TotalInitTime)
	fmt.Printf("  Time to first token: %v\n", b.TimeToFirstToken)
	fmt.Printf("  Prefill turns: %d   Decode turns: %d\n", b.PrefillTurns, b.DecodeTurns)
	for i, tps := range b.PrefillTokensPerSec {
		fmt.Printf("  Prefill turn %d: %.1f tok/s (%d tokens)\n",
			i, tps, b.PrefillTokenCounts[i])
	}
	for i, tps := range b.DecodeTokensPerSec {
		fmt.Printf("  Decode  turn %d: %.1f tok/s (%d tokens)\n",
			i, tps, b.DecodeTokenCounts[i])
	}
	fmt.Println()
}

func printCBenchmarkInfo(bi litertlm.BenchmarkInfo) {
	fmt.Println("--- BenchmarkInfo (C handle) ---")
	fmt.Printf("  Time to first token: %.3fs\n", bi.TimeToFirstToken())
	fmt.Printf("  Prefill turns: %d   Decode turns: %d\n",
		bi.NumPrefillTurns(), bi.NumDecodeTurns())
	for i := 0; i < bi.NumPrefillTurns(); i++ {
		fmt.Printf("  Prefill turn %d: %.1f tok/s (%d tokens)\n",
			i, bi.PrefillTokensPerSec(i), bi.PrefillTokenCount(i))
	}
	for i := 0; i < bi.NumDecodeTurns(); i++ {
		fmt.Printf("  Decode  turn %d: %.1f tok/s (%d tokens)\n",
			i, bi.DecodeTokensPerSec(i), bi.DecodeTokenCount(i))
	}
}
