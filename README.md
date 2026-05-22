# litertlm-go

[![Go Reference](https://pkg.go.dev/badge/github.com/vladimirvivien/litertlm-go.svg)](https://pkg.go.dev/github.com/vladimirvivien/litertlm-go)

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

LiteRT-LM does not ship prebuilt C API shared libraries. Build them
using the platform guide:

- Linux / macOS — [`LITERTLM-BUILD.md`](./LITERTLM-BUILD.md)
- Windows — [`LITERTLM-BUILD-WINDOWS.md`](./LITERTLM-BUILD-WINDOWS.md)

Minimum supported upstream: **LiteRT-LM v0.12.0**. Earlier builds lack the `optional_args` parameter on `litert_lm_conversation_send_message` / `_stream`.

## Examples

See [`examples/README.md`](./examples/README.md) for `litertlm-go` API examples.

## Documentation

- [Getting started](https://vladimirvivien.github.io/litertlm-go/getting-started/)
- [Client](https://vladimirvivien.github.io/litertlm-go/client/) — `New`, `Generate`, `GenerateStream`, `GenerateResponse`
- [Chat](https://vladimirvivien.github.io/litertlm-go/chat/) — multi-turn, system prompts, tool calling
- [Structured output](https://vladimirvivien.github.io/litertlm-go/structured-output/) — `GenerateData[T]`
- [Low-level API](https://vladimirvivien.github.io/litertlm-go/low-level/) — when to drop down
- [Troubleshooting](https://vladimirvivien.github.io/litertlm-go/troubleshooting/)

## License

Apache-2.0, same as LiteRT-LM itself.
