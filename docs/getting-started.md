# Getting started

A first run from scratch: install the Go module, point it at a built
shared library and a model file, run `Generate`.

## Prerequisites

1. **Native shared library** — LiteRT-LM doesn't distribute a
   prebuilt C API. Build it yourself following
   [`LITERTLM-BUILD.md`](https://github.com/vladimirvivien/litertlm-go/blob/main/LITERTLM-BUILD.md)
   (Linux/macOS) or
   [`LITERTLM-BUILD-WINDOWS.md`](https://github.com/vladimirvivien/litertlm-go/blob/main/LITERTLM-BUILD-WINDOWS.md).
   At the end you should have a directory containing
   `liblitertlm_c_cpu.so` (or `.dylib`/`.dll`) plus
   `libGemmaModelConstraintProvider.*` and the optional GPU
   accelerator plugins.

2. **A `.litertlm` model file** — download from Hugging Face's
   [LiteRT Community](https://huggingface.co/litert-community).
   `gemma-4-E2B-it.litertlm` is a good starting point; `gemma-4-E4B-it`
   is more capable but larger.

3. **Go 1.26 or newer**.

## Install

```bash
go mod init example.com/litertlm-demo
go get github.com/vladimirvivien/litertlm-go@latest
```

## First program

Save as `main.go`:

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

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
LITERTLM_MODEL=/abs/path/to/gemma-4-E2B-it.litertlm \
    go run main.go
```

Expected output: a short haiku from the model, e.g.

```
Blue waves crash on shore,
Salt wind whispers secrets deep,
Ocean calls to soul.
```

## What `New` does

`litertlm.New` aggregates several low-level steps:

1. Loads the shared library set (calls `Load` internally).
2. Builds an `EngineSettings` with the supplied `WithX` options.
3. Constructs an `Engine` from those settings.
4. Returns a `*Client` that owns all three handles.

`client.Close()` reverses the construction: deletes the engine and
settings handles. The shared library stays loaded for the life of the
process — purego doesn't expose `dlclose`, and the C runtime's
lifetime is tied to the process anyway.

## Defaults

When no options override them, `New` uses:

- **backend** — `cpu`
- **max tokens** — 4096 (engine total budget; prompt + output)
- **log level** — `LogError` (silences the C side's INFO/WARN chatter)
- **`WithLib("")`** falls back to `$LITERTLM_LIB`
- **`WithModel("")`** falls back to `$LITERTLM_MODEL`

For finer control — every `EngineSettings` setter has a matching
`WithX` option — see the [Client guide](client.md).

## What next

- **One-shot text generation** with sampler / token tuning →
  [Client](client.md)
- **Multi-turn conversations** with system prompts and tools →
  [Chat](chat.md)
- **Type-safe structured output** (model returns JSON, you get a
  populated struct) → [Structured output](structured-output.md)
- **Drop down to the C-API-mirroring surface** for prefill/decode,
  scoring, token introspection → [Low-level API](low-level.md)
- **Hit a quirky model output, empty completion, or stray
  `default.profraw` files?** → [Troubleshooting](troubleshooting.md)
