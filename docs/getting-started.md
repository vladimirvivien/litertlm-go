# Getting started

## Prerequisites

1. **Shared library** — LiteRT-LM doesn't distribute a
   prebuilt C API. You will need to build it yourself following instructions in
   [`LITERTLM-BUILD.md`](https://github.com/vladimirvivien/litertlm-go/blob/main/LITERTLM-BUILD.md)
   (Linux/macOS) or
   [`LITERTLM-BUILD-WINDOWS.md`](https://github.com/vladimirvivien/litertlm-go/blob/main/LITERTLM-BUILD-WINDOWS.md) (Windows).
   At the end you should have a directory containing your library dependencies.

2. **A `.litertlm` model file** — download a model file from Hugging Face's
   [LiteRT Community](https://huggingface.co/litert-community).
   `gemma-4-E2B-it.litertlm` and `gemma-4-E4B-it.litertlm` are good starting points.

3. **Go 1.26 or newer**.


## First program

Install the `litertlm-go` package:

```bash
go get github.com/vladimirvivien/litertlm-go@latest
```

Use `litertlm-go` to instantiate a LitRT-LM engine client as
shown in the snippet below:

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

`litertlm.New` aggregates several low-level steps:

1. Loads the shared library set (calls `Load` internally).
2. Builds an `EngineSettings` with the supplied options.
3. Constructs a LiteRT-LM `Engine` from those settings.
4. Returns a `*Client` with access to the engine.

Calling `client.Close()` releases internal resources and deletes the engine instance.

### Runing the program
When executing the program, specify the location of the 
share libraries and a model file:

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
LITERTLM_MODEL=/abs/path/to/gemma-4-E2B-it.litertlm \
    go run main.go
```

Expected output: a short haiku from the model:

```
Blue waves crash on shore,
Salt wind whispers secrets deep,
Ocean calls to soul.
```

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
