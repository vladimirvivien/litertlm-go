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
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.16.0)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries")
	prompt := flag.String("prompt", "Write a short poem about coding in Go.", "prompt text")
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

	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	// We benchmark 1, 2, and 4 threads
	threadCounts := []int{1, 2, 4}

	for _, threads := range threadCounts {
		fmt.Printf("\n--- Initializing Client with %d threads ---\n", threads)

		startInit := time.Now()
		client, err := litertlm.New(ctx,
			litertlm.WithLib(resolvedLib),
			litertlm.WithModel(resolvedModel),
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
