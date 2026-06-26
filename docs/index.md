# litertlm-go

A Go binding for Google's
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) for high-performance local on-device LLM inference.

> Inspired by Hybridgroup's [Yzma](https://github.com/hybridgroup/yzma).

📖 **Full documentation:**
[vladimirvivien.github.io/litertlm-go](https://vladimirvivien.github.io/litertlm-go/)

## Features

* 💬 **[Stateful Chat & Conversations](chat.md)** — Multi-turn chat orchestration with system prompts, message history, and token tracking.
* 🖼️ **[Multimodal Inputs](client.md#multimodal-inputs)** — Process text, images, and audio inputs in any order using a unified, easy-to-use interface.
* 🛠️ **[Automated Tool Calling](tools.md)** — Register standard Go functions as tools that the model can call automatically and in parallel.
* 🎯 **[Structured JSON Output](structured-output.md)** — Extract model outputs directly into Go structs with type-safe generic helpers.
* ⚡ **High Performance & GPU Acceleration** — Run on CPU or GPU (WebGPU/Direct3D 12) with built-in memory safety.
* 🔍 **[Low-Level Control](low-level.md)** — Access the full LiteRT-LM C-API when you need custom inference loops, scoring, or custom memory management.
* 🤖 **[Supported Models](models.md)** — Fully compatible with Gemma 4, Gemma 3/3n, Qwen 3/2.5, Phi-4, and other LiteRT-LM families.

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

Minimum supported upstream: **LiteRT-LM v0.13.1**.

## Examples

Self-contained programs covering every public API surface live under
[`examples/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples).
See the
[examples index](https://github.com/vladimirvivien/litertlm-go/blob/main/examples/README.md)
for the full list.

## License

Apache-2.0, same as LiteRT-LM itself.
