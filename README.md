# litertlm-go

A Go binding for Google's
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) for high-performance local on-device LLM inference.

> Inspired by Hybridgroup's [Yzma](https://github.com/hybridgroup/yzma).

📖 **Full documentation:**
[vladimirvivien.github.io/litertlm-go](https://vladimirvivien.github.io/litertlm-go/)

## Features

- **`Client`** — single-call `Generate`, range-over-func `GenerateStream`,
  rich `GenerateResponse`, plus the `*Multi` variants for image and
  audio inputs (`Text` / `Image` / `ImageFromFile` / `Audio` Parts).
  Functional options for every engine setting. Context-driven
  cancellation. → [Client guide](client.md)
- **`Chat`** — multi-turn conversations with system prompts, tool
  declarations, and structured `tool_calls` parsing.
  → [Chat guide](chat.md)
- **`GenerateData[T]` / `GenerateDataMulti[T]`** — generic helpers
  that return `*T` populated from the model's JSON output, with retry
  and tolerant parsing. The `Multi` variant accepts image and audio
  Parts. → [Structured output](structured-output.md)
- **Low-level API** — every C-API symbol exposed as a Go method.
  Useful when you need explicit prefill→decode, scoring, token
  introspection, or deterministic resource lifetimes.
  → [Low-level guide](low-level.md)


## Install

```bash
go get github.com/vladimirvivien/litertlm-go@latest
```

## Quickstart

```go
package main

import (
    "context"
    "fmt"
    "os"

    "github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
    ctx := context.Background()
    client, err := litertlm.New(ctx,
        litertlm.WithLib(os.Getenv("LITERTLM_LIB")),
        litertlm.WithModel(os.Getenv("LITERTLM_MODEL")),
    )
    if err != nil {
        fmt.Println(err)
        os.Exit(1)
    }
    defer client.Close()

    text, err := client.Generate(ctx, "Write a haiku about the sea.")
    if err != nil {
        fmt.Println(err)
        os.Exit(1)
    }
    fmt.Println(text)
}
```

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
LITERTLM_MODEL=/abs/path/to/gemma-4-E2B-it.litertlm \
    go run main.go
```

Full walkthrough → [docs/getting-started](https://vladimirvivien.github.io/litertlm-go/getting-started/).

## Building the C library

Currently, LiteRT-LM doesn't ship a prebuilt C API shared libraries. 
The followings walk you through how to build them:

- Linux / macOS — [`LITERTLM-BUILD.md`](./LITERTLM-BUILD.md)
- Windows — [`LITERTLM-BUILD-WINDOWS.md`](./LITERTLM-BUILD-WINDOWS.md)

## Examples

| Path                          | What it shows                                                      |
|-------------------------------|--------------------------------------------------------------------|
| `examples/hello/`             | Minimal `Generate`                                                 |
| `examples/stream/`            | `GenerateStream` with range-over-func                              |
| `examples/chat/`              | Multi-turn `Chat` with a system prompt                             |
| `examples/chat-history/`      | Seed `Chat` with a prior transcript via `WithInitialMessages` + `WithExtraContext` / `WithFilterChannelContentFromKVCache` / `WithMaxToolHops` |
| `examples/conversation/`      | `Chat` + `NewRawTool` + manual dispatch via `Reply.ToolCalls`      |
| `examples/autotool/`          | `Chat` + `RegisterTool` + auto-dispatch                            |
| `examples/tool-policy/`       | `WithToolPolicy(ToolPolicyReturnOnError vs ToolPolicyInformOnError)` — handler-error behavior under auto-dispatch |
| `examples/structured/`        | `GenerateData[T]` (typed JSON output via reflection)               |
| `examples/vision/`            | `GenerateMulti` (image + text) with self-comparison against a sidecar |
| `examples/audio/`             | `GenerateMulti` (audio + text) — transcription with optional alignment vs a reference |
| `examples/extract/`           | `GenerateDataMulti[T]` (image-to-typed-JSON) with self-comparison  |
| `examples/cancel/`            | Cancelling a streaming generation via `context.WithCancel`         |
| `examples/prefill-decode/`    | Explicit two-phase generation (low-level)                          |
| `examples/conversation-lowlevel/` | Low-level twin of Chat: hand-built `SessionConfig` + `ConversationConfig` + `SendMessage` + `RenderMessage` + `BenchmarkInfo` |
| `examples/score/`             | `ScoreTexts` + `Score` / `TokenLength` (low-level)                 |
| `examples/token-scores/`      | `ScoreTexts` + `TokenScores` per-token log-probs paired with `Engine.Tokenize` |
| `examples/tokenize/`          | `Engine.Tokenize` / `Detokenize` + start/stop tokens (low-level)   |
| `examples/gpu/`               | GPU-backed generation                                              |
| `examples/benchmarks/`        | Per-generation benchmark capture via `WithBenchmarkEnabled` + `Response.Benchmark()` |
| `examples/cache-warmup/`      | Cold-vs-warm `WithCacheDir` load — XNNPACK / mldrift artefact reuse        |
| `examples/activation-dtype/`  | Default-vs-selected `WithActivationDataType` (F32 / F16 / I16 / I8) — empirical per-backend deltas |
| `examples/prefill-chunk/`     | Default-vs-selected `WithPrefillChunkSize` (CPU-only) — chunked vs unchunked prefill timings |
| `examples/parallel-load/`     | Parallel vs serial `WithParallelSectionLoading` — `litertlm.New` wall-clock delta |
| `examples/logging/`           | `SetMinLogLevel` — set the LiteRT-LM log severity floor at startup and toggle mid-program |
| `examples/per-call-sampler/`  | `WithSampler` per-call override — three sampler shapes (Deterministic / Balanced / Creative) on the same Client |
| `examples/speculative/`       | Side-by-side throughput comparison with / without `WithSpeculativeDecodingEnabled` |

## Documentation

- [Getting started](https://vladimirvivien.github.io/litertlm-go/getting-started/)
- [Client](https://vladimirvivien.github.io/litertlm-go/client/) — `New`, `Generate`, `GenerateStream`, `GenerateResponse`
- [Chat](https://vladimirvivien.github.io/litertlm-go/chat/) — multi-turn, system prompts, tool calling
- [Structured output](https://vladimirvivien.github.io/litertlm-go/structured-output/) — `GenerateData[T]`
- [Low-level API](https://vladimirvivien.github.io/litertlm-go/low-level/) — when to drop down
- [Troubleshooting](https://vladimirvivien.github.io/litertlm-go/troubleshooting/)

## License

Apache-2.0, same as LiteRT-LM itself.
