# litertlm-go

A Go wrapper for Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) C API for running
local LLM inference.

`litertlm-go` uses `ebitengine/purego` to 
dynamically load the LiteRT-LM C API shared library at runtime.
No CGo toolchain is required to build applications with this package.
Note: this approach was inspired by project Hybridgroup's project [Yzma](https://github.com/hybridgroup/yzma).

## Building LiteRT-LM C shared object libraries
Project LiteRT-LM is a C++ projects and does not distribute a C API by default.
So, if you want to use LiteRT-LM locally for inference in Go, you must first compile the
shared librariries to expose a C API.  

Folllow instructions [here](./build_litertlm.md) to build the the shared object binaries.

## Install

```bash
go get github.com/vladimirvivien/litertlm-go@latest
```

### Model files
You will need to download the `*.litertlm` model
that you want to use for inference. You can get the models from Hugging Face's 
[LiteRT Community](https://huggingface.co/litert-community). For
the example below, we will use `litert-community/gemma-4-E2B-it-litert-lm`.

## Using `litertlm-go`

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

Run the code with:

```bash
LITERTLM_LIB=/path/to/shared-objects/lib \
    go run ./examples/hello -model /abs/path/to/model.litertlm
```

## Examples

| Path                 | What it shows                                                      |
|----------------------|--------------------------------------------------------------------|
| `examples/hello/`    | Minimal synchronous generation                                     |
| `examples/stream/`   | Token-by-token streaming using the Go channel variant              |
| `examples/chat/`     | Multi-turn Conversation API with JSON messages                     |
| `examples/gpu/`      | GPU-backed generation + BenchmarkInfo metrics                      |



## License

Apache-2.0, same as LiteRT-LM itself.
