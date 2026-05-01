# litertlm-go

A purego-backed, cgo-free Go wrapper for Google's
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) C API.
Run local LLM inference from Go without a C toolchain — the C library
is loaded dynamically at runtime via
[`ebitengine/purego`](https://github.com/ebitengine/purego).

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
        panic(err)
    }
    defer client.Close()

    text, err := client.Generate(ctx, "Write a haiku about the sea.")
    if err != nil {
        panic(err)
    }
    fmt.Println(text)
}
```

Run it:

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
LITERTLM_MODEL=/abs/path/to/gemma-4-E2B-it.litertlm \
    go run main.go
```

Full walkthrough: [Getting started](getting-started.md).

## What's in the box

- **`Client`** — single-call `Generate`, range-over-func `GenerateStream`,
  rich `GenerateResponse`. Functional options for every engine setting.
  Context-driven cancellation. → [Client guide](client.md)
- **`Chat`** — multi-turn conversations with system prompts, tool
  declarations, and structured `tool_calls` parsing.
  → [Chat guide](chat.md)
- **`GenerateData[T]`** — generic helper that returns `*T` populated
  from the model's JSON output, with retry and tolerant parsing.
  → [Structured output](structured-output.md)
- **Low-level API** — every C-API symbol exposed as a Go method.
  Useful when you need explicit prefill→decode, scoring, token
  introspection, or deterministic resource lifetimes.
  → [Low-level guide](low-level.md)

## Building the C library

LiteRT-LM doesn't ship a prebuilt C API. Build it yourself:
[`LITERTLM-BUILD.md`](https://github.com/vladimirvivien/litertlm-go/blob/main/LITERTLM-BUILD.md)
(Linux/macOS) or
[`LITERTLM-BUILD-WINDOWS.md`](https://github.com/vladimirvivien/litertlm-go/blob/main/LITERTLM-BUILD-WINDOWS.md).

## Examples

A runnable demo lives in `examples/<name>/` for each major API:

| Example                    | What it shows                                                  |
|----------------------------|----------------------------------------------------------------|
| `examples/hello/`          | Minimal `Generate`                                             |
| `examples/stream/`         | `GenerateStream` with range-over-func                          |
| `examples/chat/`           | Multi-turn `Chat` with a system prompt                         |
| `examples/conversation/`   | `Chat` + tools + structured `tool_calls`                       |
| `examples/structured/`     | `GenerateData[Recipe]` with retries                            |
| `examples/cancel/`         | Mid-stream cancel via `context.WithCancel`                     |
| `examples/prefill-decode/` | Explicit two-phase generation (low-level)                      |
| `examples/score/`          | `Session.ScoreTexts` + `Score` / `TokenLength` (low-level)     |
| `examples/tokenize/`       | `Engine.Tokenize` / `Detokenize` + start/stop tokens (low-level) |
| `examples/gpu/`            | GPU backend + benchmark metrics                                |

## License

Apache-2.0, same as LiteRT-LM itself.
