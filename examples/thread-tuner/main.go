// thread-tuner runs a quick benchmark prompt across different CPU thread counts
// to demonstrate how allocating thread resources affects generation speed.
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
	model := flag.String("model", "", "path to .litertlm model file")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries")
	prompt := flag.String("prompt", "Write a short poem about coding in Go.", "prompt text")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "--model is required")
		os.Exit(2)
	}

	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	ctx := context.Background()

	// We benchmark 1, 2, and 4 threads
	threadCounts := []int{1, 2, 4}

	for _, threads := range threadCounts {
		fmt.Printf("\n--- Initializing Client with %d threads ---\n", threads)

		startInit := time.Now()
		client, err := litertlm.New(ctx,
			litertlm.WithLib(*libPath),
			litertlm.WithModel(*model),
			litertlm.WithBackend("cpu"),
			litertlm.WithNumThreads(threads),
			litertlm.WithBenchmarkEnabled(),
		)
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to create client: %v\n", err)
			continue
		}

		startGen := time.Now()
		res, err := client.GenerateResponse(ctx, *prompt, litertlm.WithMaxOutputTokens(64))
		if err != nil {
			fmt.Fprintf(os.Stderr, "generate failed: %v\n", err)
			client.Close()
			continue
		}
		genDuration := time.Since(startGen)
		initDuration := time.Since(startInit) - genDuration

		b := res.Benchmark()
		var tokensPerSec float64
		var totalTokens int
		if b != nil {
			for _, c := range b.PrefillTokenCounts {
				totalTokens += c
			}
			for _, c := range b.DecodeTokenCounts {
				totalTokens += c
			}
			tokensPerSec = float64(totalTokens) / genDuration.Seconds()
		}

		fmt.Printf("Text generated: %q\n", res.Text())
		fmt.Printf("Init time:      %v\n", initDuration)
		fmt.Printf("Inference time: %v\n", genDuration)
		if b != nil {
			fmt.Printf("Throughput:     %.2f tokens/sec (total %d tokens)\n", tokensPerSec, totalTokens)
		}

		client.Close()
	}
}
