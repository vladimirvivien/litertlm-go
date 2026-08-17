// prefill-chunk contrasts the engine's default prefill chunking with
// a caller-selected WithPrefillChunkSize. Both runs are cold — no
// internal warmup pass — to match upstream
// litert_lm_advanced_main --benchmark methodology.
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

const defaultPrompt = `The printing press, developed by Johannes Gutenberg in the mid-fifteenth century, transformed European intellectual life by making books cheaper and faster to produce than scribal copying. Before the press, every manuscript was hand-copied, a process that could take a single scribe months or even years to complete for a single volume. The cost of a hand-copied Bible was equivalent to several years of an average craftsman's wages. Gutenberg's contribution was not the invention of movable type, which already existed in China and Korea, but rather his combination of several existing technologies into a practical workflow. He developed a hand mould that produced type pieces of consistent dimensions, a press adapted from those used to extract olive oil and wine, and an oil-based ink that adhered well to metal type and transferred cleanly to paper. The first major work printed using his system was a Latin Bible completed around 1454. Within fifty years of its appearance, printing presses had spread to most major European cities, and an estimated twenty million volumes had been produced. This rapid expansion changed how knowledge circulated. Standardized texts could be distributed widely, reducing transcription errors that had accumulated through generations of copying. Authors could be reasonably sure that readers in distant cities were seeing the same words they had written, which made cross-reference and citation practical. Scientific and scholarly work began to build on shared corpora rather than on the local manuscripts available at a particular monastery or university. The reproducibility of printed matter also altered the relationship between texts and authority. Errors in early printed editions, once recognized, could be corrected in subsequent runs, but in the interim they propagated faster than scribal mistakes ever had. Religious authorities discovered that vernacular translations of scripture could no longer be controlled through manuscript regulation, a fact that would shape the Reformation half a century later. Educational practices shifted as well. Standardized textbooks made it possible for universities to coordinate curriculum across institutions, and the lower cost of printed material meant that students could own personal copies of works they had previously consulted only in chained library collections. Please summarize this paragraph in two sentences.`

func main() {
	model := flag.String("model", "", "path to .litertlm model file (required)")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.16.0)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu); WithPrefillChunkSize is CPU-only")
	chunk := flag.Int("chunk", 128, "prefill chunk size for run 2; -1 disables chunking")
	prompt := flag.String("prompt", defaultPrompt, "long prompt that makes prefill dominate the Generate timing")
	cacheDir := flag.String("cache-dir", "", "directory passed to WithCacheDir; empty creates a fresh temp dir removed at exit")
	keepCache := flag.Bool("keep-cache", false, "skip cleanup of an auto-created temp cache dir")
	maxTokens := flag.Int("max-tokens", 4096, "engine total token budget (prompt + output)")
	flag.Parse()

	ctx := context.Background()

	resolvedLib := *libPath
	if *getLib != "" {
		staged, err := litertlm.FetchLib("", "", *getLib)
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

	if *chunk == 0 {
		fmt.Fprintln(os.Stderr, "--chunk must be a positive integer or -1 (disable)")
		os.Exit(2)
	}
	if *backend != "cpu" {
		fmt.Printf("note: -backend %q ignores WithPrefillChunkSize (CPU-only)\n", *backend)
	}

	dir, cleanup, err := resolveCacheDir(*cacheDir, *keepCache)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cache dir: %v\n", err)
		os.Exit(1)
	}
	defer cleanup()
	fmt.Printf("cache dir: %s\n", dir)

	fmt.Println("\n=== Run 1: default (no WithPrefillChunkSize) ===")
	def, err := runOnce(ctx, resolvedLib, resolvedModel, *backend, dir, nil, *maxTokens, *prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "run 1: %v\n", err)
		os.Exit(1)
	}
	printRun(def)

	fmt.Printf("\n=== Run 2: WithPrefillChunkSize(%d) ===\n", *chunk)
	sel, err := runOnce(ctx, resolvedLib, resolvedModel, *backend, dir, chunk, *maxTokens, *prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "run 2: %v\n", err)
		os.Exit(1)
	}
	printRun(sel)

	fmt.Println("\n=== Delta (run 2 - run 1) ===")
	fmt.Printf("  TimeToFirstToken: %v\n", sel.bench.TimeToFirstToken-def.bench.TimeToFirstToken)
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
	d, err := os.MkdirTemp("", "litertlm-prefill-chunk-*")
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

func runOnce(ctx context.Context, libPath, modelPath, backend, cacheDir string, chunk *int, maxTokens int, prompt string) (*runResult, error) {
	opts := []litertlm.Option{
		litertlm.WithLib(libPath),
		litertlm.WithModel(modelPath),
		litertlm.WithBackend(backend),
		litertlm.WithCacheDir(cacheDir),
		litertlm.WithMaxTokens(maxTokens),
		litertlm.WithBenchmarkEnabled(),
	}
	if chunk != nil {
		opts = append(opts, litertlm.WithPrefillChunkSize(*chunk))
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
	if r.bench == nil {
		return
	}
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
