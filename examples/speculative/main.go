// speculative compares decode throughput across SpeculativeDecodingOff,
// SpeculativeDecodingOn, and SpeculativeDecodingAdaptive (with optional
// draft acceptance lenience) using WithSpeculativeDecoding.
//
// Requires a model that includes a speculative decoding drafter, such as
// Gemma 4 E4B.
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
	label    string
	wallTime time.Duration
	bench    *litertlm.Benchmark
	text     string
}

func main() {
	model := flag.String("model", "", "path to a .litertlm model that supports speculative decoding (e.g. Gemma 4)")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.19.1)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	prompt := flag.String("prompt", "Explain how speculative decoding accelerates autoregressive LLM inference.", "prompt text")
	lenience := flag.Float64("lenience", 0, "speculative acceptance lenience in [0, 1] (0 = lossless)")
	maxOutput := flag.Int("max-output", 128, "max output tokens per call")
	flag.Parse()

	ctx := context.Background()

	resolvedLib := *libPath
	if *getLib != "" {
		staged, err := litertlm.FetchLib(ctx, "", "", *getLib)
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

	modes := []struct {
		label string
		mode  litertlm.SpeculativeDecodingMode
	}{
		{"Off (baseline)", litertlm.SpeculativeDecodingOff},
		{"On", litertlm.SpeculativeDecodingOn},
		{"Adaptive", litertlm.SpeculativeDecodingAdaptive},
	}

	results := make([]runResult, 0, len(modes))
	for _, entry := range modes {
		fmt.Printf("=== Mode: %s (lenience=%.2f) ===\n", entry.label, *lenience)
		r, err := runOnce(ctx, resolvedLib, resolvedModel, *backend, *prompt, *maxOutput, litertlm.SpeculativeDecodingConfig{
			Mode:     entry.mode,
			Lenience: float32(*lenience),
		})
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s run: %v\n", entry.label, err)
			os.Exit(1)
		}
		r.label = entry.label
		fmt.Println(r.text)
		printBenchmark(r, "  ")
		fmt.Println()
		results = append(results, r)
	}

	if len(results) > 1 {
		fmt.Println("=== Comparison vs Off ===")
		base := results[0]
		for _, r := range results[1:] {
			printSpeedup(base, r)
		}
	}
}

func runOnce(ctx context.Context, libPath, model, backend, prompt string, maxOutput int, spec litertlm.SpeculativeDecodingConfig) (runResult, error) {
	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	client, err := litertlm.New(ctx,
		litertlm.WithLib(libPath),
		litertlm.WithModel(model),
		litertlm.WithBackend(backend),
		litertlm.WithBenchmarkEnabled(),
		litertlm.WithSpeculativeDecoding(spec),
	)
	if err != nil {
		return runResult{}, fmt.Errorf("new client: %w", err)
	}
	defer func() { _ = client.Close() }()

	chat, err := client.NewChat(ctx)
	if err != nil {
		return runResult{}, fmt.Errorf("new chat: %w", err)
	}
	defer func() { _ = chat.Close() }()

	start := time.Now()
	reply, err := chat.Send(ctx, prompt, litertlm.WithMaxOutputTokens(maxOutput))
	wall := time.Since(start)
	if err != nil {
		return runResult{}, fmt.Errorf("chat send: %w", err)
	}
	bi, err := chat.Conversation().BenchmarkInfo()
	if err != nil {
		return runResult{}, fmt.Errorf("benchmark info: %w", err)
	}
	defer bi.Delete()

	bench := &litertlm.Benchmark{
		TimeToFirstToken: time.Duration(bi.TimeToFirstToken() * float64(time.Second)),
		TotalInitTime:    time.Duration(bi.TotalInitTime() * float64(time.Second)),
		MaxNumTokens:     bi.MaxNumTokens(),
		PrefillTurns:     bi.NumPrefillTurns(),
		DecodeTurns:      bi.NumDecodeTurns(),
	}
	for i := range bench.PrefillTurns {
		bench.PrefillTokenCounts = append(bench.PrefillTokenCounts, bi.PrefillTokenCount(i))
		bench.PrefillTokensPerSec = append(bench.PrefillTokensPerSec, bi.PrefillTokensPerSec(i))
	}
	for i := range bench.DecodeTurns {
		bench.DecodeTokenCounts = append(bench.DecodeTokenCounts, bi.DecodeTokenCount(i))
		bench.DecodeTokensPerSec = append(bench.DecodeTokensPerSec, bi.DecodeTokensPerSec(i))
	}
	return runResult{wallTime: wall, bench: bench, text: reply.Text()}, nil
}

func printBenchmark(r runResult, indent string) {
	b := r.bench
	fmt.Printf("%sInit time:           %v\n", indent, b.TotalInitTime)
	fmt.Printf("%sMax KV tokens:       %d\n", indent, b.MaxNumTokens)
	fmt.Printf("%sTime to first token: %v\n", indent, b.TimeToFirstToken)
	fmt.Printf("%sWall-clock send:     %v\n", indent, r.wallTime)
	for i, tps := range b.PrefillTokensPerSec {
		fmt.Printf("%sPrefill turn %d:      %.1f tok/s  (%d tokens)\n",
			indent, i, tps, b.PrefillTokenCounts[i])
	}
	for i, tps := range b.DecodeTokensPerSec {
		fmt.Printf("%sDecode  turn %d:      %.1f tok/s  (%d tokens)\n",
			indent, i, tps, b.DecodeTokenCounts[i])
	}
}

func printSpeedup(base, cand runResult) {
	baseTPS := avg(base.bench.DecodeTokensPerSec)
	candTPS := avg(cand.bench.DecodeTokensPerSec)
	fmt.Printf("  %-14s decode: %.1f tok/s", cand.label, candTPS)
	if baseTPS > 0 {
		fmt.Printf(" (%.2fx)", candTPS/baseTPS)
	}
	if cand.wallTime > 0 && base.wallTime > 0 {
		fmt.Printf(" | wall: %v (%.2fx)", cand.wallTime, float64(base.wallTime)/float64(cand.wallTime))
	}
	fmt.Println()
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
