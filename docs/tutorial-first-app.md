# Your First Local LLM App in 5 Minutes

Running an open-weights model directly inside a Go process removes the operational overhead of managing external inference servers or separate container runtimes. In this tutorial, you build a complete Go command-line program that fetches the LiteRT-LM shared library, downloads a compact Gemma 3 model, inspects its capabilities before loading weights, streams tokens live to standard output, and reports per-turn throughput benchmarks.

<div class="lrt-meta-bar">
  <span class="lrt-pill lrt-pill--coral">~5 min</span>
  <span class="lrt-pill lrt-pill--teal">Text &amp; Streaming</span>
  <span class="lrt-pill lrt-pill--gold">gemma3-1b-it-int4</span>
  <span class="lrt-pill lrt-pill--plum">CPU or GPU</span>
</div>

---

## Step 1 — Create Your Go Module

Start from an empty project directory, initialize a Go module, and add `litertlm-go` as a dependency:

```bash
mkdir local-llm-demo
cd local-llm-demo
go mod init local-llm-demo
go get github.com/vladimirvivien/litertlm-go@latest
```

---

## Step 2 — Fetch Runtime Binaries & Model Weights Automatically

Instead of asking users to download zip archives and export environment paths beforehand, your program can call `litertlm.FetchLib` and `litertlm.FetchModel`. Both helpers verify checksums and store artifacts in your user cache directory so subsequent runs start immediately offline.

Attach `litertlm.WithModelProgress` to render clean download progress while weights stream from Hugging Face:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"runtime"
	"time"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	ctx := context.Background()

	// 1. Download and cache official LiteRT-LM shared libraries for this host.
	libDir, err := litertlm.FetchLib(ctx, runtime.GOOS, runtime.GOARCH, "v0.19.1")
	if err != nil {
		log.Fatalf("FetchLib failed: %v", err)
	}

	// 2. Download and cache Gemma 3 1B (quantized int4) from Hugging Face.
	modelPath, err := litertlm.FetchModel(ctx, "litert-community/gemma3-1b-it-int4",
		litertlm.WithModelProgress(func(downloaded, total int64, pct float64) {
			if total > 0 {
				fmt.Printf("\rDownloading model: %.1f%% (%.1f / %.1f MB)",
					pct, float64(downloaded)/1e6, float64(total)/1e6)
			}
		}),
	)
	if err != nil {
		log.Fatalf("FetchModel failed: %v", err)
	}
	fmt.Println()
```

---

## Step 3 — Inspect Model Capabilities Before Booting

Large `.litertlm` files embed capability metadata—context limits, supported modalities (text, vision, audio), thinking support, tool-calling compatibility, and default sampling settings. Calling `litertlm.Load` followed by `litertlm.InspectModel` reads that header without allocating tensor weights or KV caches, making it ideal for startup validation:

```go
	if err := litertlm.Load(libDir, "cpu", ""); err != nil {
		log.Fatalf("Load failed: %v", err)
	}
	maj, minVer, pat, err := litertlm.APIVersion()
	if err != nil {
		log.Fatalf("APIVersion failed: %v", err)
	}
	caps, err := litertlm.InspectModel(modelPath)
	if err != nil {
		log.Fatalf("InspectModel failed: %v", err)
	}
	fmt.Printf("LiteRT-LM C API %d.%d.%d | type=%s max_ctx=%d vision=%v thinking=%v tools=%v\n",
		maj, minVer, pat, caps.ModelType, caps.MaxContextTokens,
		caps.SupportsVision(), caps.SupportsThinking, caps.SupportsFunctionCalling)
```

---

## Step 4 — Stream Tokens & Measure Throughput

Initialize a `Client` with `litertlm.WithBenchmarkEnabled()` so the runtime records time-to-first-token (TTFT) and prefill/decode speeds. Always defer `client.Close()` to release native engine handles cleanly when the function returns.

Pair `context.WithTimeout` with `client.GenerateStream` (which returns a Go `iter.Seq2[litertlm.Chunk, error]` iterator) to print each token as soon as the model decodes it, then run `client.GenerateResponse` when you want structured benchmark numbers alongside the reply:

```go
	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	client, err := litertlm.New(ctx,
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
		litertlm.WithBackend("cpu"),
		litertlm.WithMaxTokens(2048),
		litertlm.WithBenchmarkEnabled(),
	)
	if err != nil {
		log.Fatalf("New client failed: %v", err)
	}
	defer client.Close()

	runCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()

	fmt.Println("--- Streaming Reply ---")
	for chunk, err := range client.GenerateStream(runCtx, "Explain why Go uses goroutines in three sentences.") {
		if err != nil {
			log.Fatalf("GenerateStream failed: %v", err)
		}
		fmt.Print(chunk.Text)
		if chunk.Final {
			fmt.Println()
		}
	}

	fmt.Println("--- Timed Response & Benchmark ---")
	resp, err := client.GenerateResponse(runCtx, "Give one tip for writing clean concurrent Go code.")
	if err != nil {
		log.Fatalf("GenerateResponse failed: %v", err)
	}
	fmt.Println(resp.Text())

	if b := resp.Benchmark(); b != nil {
		fmt.Printf("TTFT: %v | KV Cap: %d tokens\n", b.TimeToFirstToken, b.MaxNumTokens)
		for i, tps := range b.DecodeTokensPerSec {
			fmt.Printf("Decode turn %d: %.1f tok/s (%d tokens)\n",
				i, tps, b.DecodeTokenCounts[i])
		}
	}
}
```

---

## Run & Expected Output

Run your program with a single command:

```bash
go run .
```

*(Host-specific timings and throughput numbers differ.)*

```text
LiteRT-LM C API 1.0.0 | type=LLM max_ctx=32768 vision=false thinking=false tools=true
--- Streaming Reply ---
Go uses lightweight goroutines instead of heavy operating-system threads so programs can launch thousands of concurrent tasks with minimal memory overhead. The Go runtime multiplexes goroutines onto a small pool of OS threads automatically. Paired with channels, this makes concurrent workflows simple to write and reason about.
--- Timed Response & Benchmark ---
Pass a `context.Context` into blocking or streaming calls so cancellation propagates cleanly when work is no longer needed.
TTFT: 142ms | KV Cap: 2048 tokens
Decode turn 0: 41.6 tok/s (24 tokens)
```

---

## Try the Repository Examples

Once you see this pattern working, explore these focused programs in `examples/`:

* [`examples/hello/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/hello) — minimal one-shot `Generate` CLI with `-get-lib` and `-get-model` flags.
* [`examples/stream/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/stream) — token streaming on a multi-turn `Chat` with `-system` and `-speculative` toggles.
* [`examples/capabilities/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/capabilities) — zero-allocation `.litertlm` metadata inspector with `-json` output.
* [`examples/benchmarks/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/benchmarks) — side-by-side comparison of `Response.Benchmark()` and low-level `Session.BenchmarkInfo()`.

## Next Steps

* Continue to [Vision Q&A & Structured JSON Extraction](tutorial-vision-json.md) to work with images and typed Go structs.
* Review [Client Guide](client.md) for the complete list of client construction options and backend flags.
