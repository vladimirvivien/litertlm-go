# litertlm-go

A Go wrapper for Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) C API for running
local LLM inference.

`litertlm-go` uses `ebitengine/purego` to 
dynamically load the LiteRT-LM C API shared library at runtime.
No CGo toolchain is required to build applications with this package.
Note: this approach was inspired by project Hybridgroup's project Yzma.

## Install

```bash
go get github.com/vladimirvivien/litertlm-go@latest
```

### Build C API library
You will also need the LiteRT-LM native shared library. See
[`build_litertlm.md`](./build_litertlm.md) for instructions on how to build
the all necessary dynamic library files needed to run LiteRT-LM.

### Model files
You will also need to download the `*.litertlm` version of the Gemma model
that you want to use for inference. You can get the models from Hugging Face's 
[LiteRT Community](https://huggingface.co/litert-community).

## Minimal usage

```go
package main

import (
    "fmt"
    "os"

    "github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
    if err := litertlm.Load(os.Getenv("LITERTLM_LIB")); err != nil {
        panic(err)
    }
    defer litertlm.Close()

    settings, _ := litertlm.NewEngineSettings(
        "/abs/path/to/gemma.litertlm", "cpu", nil, nil)
    defer settings.Delete()

    engine, _ := litertlm.NewEngine(settings)
    defer engine.Delete()

    session, _ := engine.NewSession(0)
    defer session.Delete()

    resp, _ := session.GenerateContent([]litertlm.InputData{
        litertlm.NewTextInputString("Write a haiku about the sea."),
    })
    defer resp.Delete()

    fmt.Println(resp.Text(0))
}
```

## Examples

| Path                 | What it shows                                                      |
|----------------------|--------------------------------------------------------------------|
| `examples/hello/`    | Minimal synchronous generation                                     |
| `examples/stream/`   | Token-by-token streaming using the Go channel variant              |
| `examples/chat/`     | Multi-turn Conversation API with JSON messages                     |
| `examples/gpu/`      | GPU-backed generation + BenchmarkInfo metrics                      |

Run any of them with:

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/hello -model /abs/path/to/model.litertlm
```

## Project layout

```
pkg/
  loader/     resolves lib<name>.{so,dylib,dll} and opens it via purego
  utils/      cross-platform string ↔ *byte helpers
  litertlm/   the Go API: bindings, Engine, Session, Conversation, ...
examples/
  hello/  stream/  chat/  gpu/
```

## License

Apache-2.0, same as LiteRT-LM itself.
