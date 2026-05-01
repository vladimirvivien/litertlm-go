# litertlm-go

A purego-backed, cgo-free Go wrapper for Google's
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) C API. Run
local LLM inference from Go without a C toolchain — the C library is
loaded dynamically at runtime via
[`ebitengine/purego`](https://github.com/ebitengine/purego).

> Inspired by Hybridgroup's [Yzma](https://github.com/hybridgroup/yzma).

📖 **Full documentation:**
[vladimirvivien.github.io/litertlm-go](https://vladimirvivien.github.io/litertlm-go/)

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

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
LITERTLM_MODEL=/abs/path/to/gemma-4-E2B-it.litertlm \
    go run main.go
```

Full walkthrough → [docs/getting-started](https://vladimirvivien.github.io/litertlm-go/getting-started/).

## Building the C library

LiteRT-LM doesn't ship a prebuilt C API. Build it yourself:

- Linux / macOS — [`LITERTLM-BUILD.md`](./LITERTLM-BUILD.md)
- Windows — [`LITERTLM-BUILD-WINDOWS.md`](./LITERTLM-BUILD-WINDOWS.md)

## Examples

| Path                          | What it shows                                                      |
|-------------------------------|--------------------------------------------------------------------|
| `examples/hello/`             | Minimal `Generate`                                                 |
| `examples/stream/`            | `GenerateStream` with range-over-func                              |
| `examples/chat/`              | Multi-turn `Chat` with a system prompt                             |
| `examples/conversation/`      | `Chat` + tools + structured tool_calls                             |
| `examples/structured/`        | `GenerateData[T]` (typed JSON output via reflection + retries)     |
| `examples/cancel/`            | Cancelling a streaming generation via `context.WithCancel`         |
| `examples/prefill-decode/`    | Explicit two-phase generation (low-level)                          |
| `examples/score/`             | `ScoreTexts` + `Score` / `TokenLength` (low-level)                 |
| `examples/tokenize/`          | `Engine.Tokenize` / `Detokenize` + start/stop tokens (low-level)   |
| `examples/gpu/`               | GPU-backed generation + benchmark metrics                          |

## Documentation

The high-level API (recommended for most use cases) is documented at
[vladimirvivien.github.io/litertlm-go](https://vladimirvivien.github.io/litertlm-go/):

- [Getting started](https://vladimirvivien.github.io/litertlm-go/getting-started/)
- [Client](https://vladimirvivien.github.io/litertlm-go/client/) — `New`, `Generate`, `GenerateStream`, `GenerateResponse`
- [Chat](https://vladimirvivien.github.io/litertlm-go/chat/) — multi-turn, system prompts, tool calling
- [Structured output](https://vladimirvivien.github.io/litertlm-go/structured-output/) — `GenerateData[T]`
- [Low-level API](https://vladimirvivien.github.io/litertlm-go/low-level/) — when to drop down
- [Troubleshooting](https://vladimirvivien.github.io/litertlm-go/troubleshooting/)

The site is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
and published from the `docs/` directory on tagged pushes (see
`.github/workflows/docs.yml`). Preview locally with
`pip install mkdocs-material && mkdocs serve`.

## License

Apache-2.0, same as LiteRT-LM itself.
