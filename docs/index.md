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
  declarations, and structured `tool_calls` parsing. Per-call
  `RuntimeOption` knobs (`WithVisualTokenBudget`,
  `WithReturnToolRequests`, `WithMaxConcurrentTools`), parallel tool
  dispatch, multimodal history seeding via `WithInitialMessages`,
  and `Chat.TokenCount()` for cumulative usage.
  → [Chat guide](chat.md)
- **`GenerateData[T]` / `GenerateDataMulti[T]`** — generic helpers
  that return `*T` populated from the model's response via a
  synthesized tool-call capture (primary path) with a
  prompt-engineered fallback. The `Multi` variant accepts image and
  audio Parts. → [Structured output](structured-output.md)
- **Low-level API** — every C-API symbol exposed as a Go method.
  Useful when you need explicit prefill→decode, scoring, token
  introspection, or deterministic resource lifetimes.
  → [Low-level guide](low-level.md)
- **Supported models** — Gemma 4 / 3 / 3n, FunctionGemma, Qwen 3 / 2.5,
  Phi-4, and other LiteRT-LM families, with the processor mapping and
  per-family caveats. → [Supported models](models.md)

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

Full walkthrough: [Getting started](getting-started.md).

## Building the C library

LiteRT-LM does not ship prebuilt C API shared libraries. Build them
using the platform guide:
[`LITERTLM-BUILD.md`](https://github.com/vladimirvivien/litertlm-go/blob/main/LITERTLM-BUILD.md)
(Linux/macOS) or
[`LITERTLM-BUILD-WINDOWS.md`](https://github.com/vladimirvivien/litertlm-go/blob/main/LITERTLM-BUILD-WINDOWS.md).

Minimum supported upstream: **LiteRT-LM v0.12.0**.

## Examples

Self-contained programs covering every public API surface live under
[`examples/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples).
See the
[examples index](https://github.com/vladimirvivien/litertlm-go/blob/main/examples/README.md)
for the full list.

## License

Apache-2.0, same as LiteRT-LM itself.
