# litertlm-go

A Go wrapper for Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) for running
local LLM inference.

`litertlm-go` uses `ebitengine/purego` to 
dynamically load the LiteRT-LM C API shared library at runtime.
No CGo toolchain is required to build applications with this package.
Note: this approach was inspired by project Hybridgroup's project [Yzma](https://github.com/hybridgroup/yzma).

## Building LiteRT-LM C shared object libraries
LiteRT-LM is a C++ project and does not distribute a C API by default.
However, you can folllow instructions [here](./LITERTLM-BUILD.md) to build 
the LiteRT-ML source code with C shared object libraries.


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
    if err := litertlm.Load(os.Getenv("LITERTLM_LIB"), "cpu"); err != nil {
        panic(err)
    }
    defer litertlm.Close()

    settings, _ := litertlm.NewEngineSettings(
        os.Getenv("LITERTLM_MODEL"), "cpu", nil, nil)
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

### Running the snippet

In a fresh directory, save the code above as `main.go`, then:

```bash

LITERTLM_LIB=/abs/path/to/dist/lib \
LITERTLM_MODEL=/abs/path/to/gemma-4-E2B-it.litertlm \
    go run main.go
```

Expected output: a short haiku written by the model, e.g.

For the full set of runnable demos see [`examples/`](#examples).

## API map

The package surface is small; this is what to reach for:

| You want to…                          | Use                                                                            |
|---------------------------------------|--------------------------------------------------------------------------------|
| Load and run a model                  | `Load`, `NewEngineSettings`, `NewEngine`, `Engine.NewSession`                  |
| One-shot generation                   | `Session.GenerateContent` ([hello](examples/hello/))                           |
| Token-by-token streaming              | `Session.GenerateContentStreamCh` ([stream](examples/stream/))                 |
| Multi-turn chat with a system prompt  | `Engine.NewConversation` + `Conversation.SendMessage` ([chat](examples/chat/)) |
| Tool-using agents                     | `NewConversationConfig` with `toolsJSON` ([conversation](examples/conversation/)) |
| Tokenize / detokenize                 | `Engine.Tokenize`, `Engine.Detokenize` ([tokenize](examples/tokenize/))        |
| Inspect model start/stop tokens       | `Engine.StartTokenIDs`, `Engine.StopTokenIDs` ([tokenize](examples/tokenize/)) |
| Manual prefill→decode                 | `Session.RunPrefill`, `Session.RunDecode` ([prefill-decode](examples/prefill-decode/)) |
| Score candidate completions           | `Session.ScoreTexts`, `Responses.Score` ([score](examples/score/))             |
| Cancel an in-flight stream            | `Session.Cancel` ([cancel](examples/cancel/))                                  |
| GPU + benchmark metrics               | `EngineSettings.EnableBenchmark`, `Session.BenchmarkInfo` ([gpu](examples/gpu/)) |

## Examples

| Path                          | What it shows                                                      |
|-------------------------------|--------------------------------------------------------------------|
| `examples/hello/`             | Minimal synchronous generation                                     |
| `examples/stream/`            | Token-by-token streaming using the Go channel variant              |
| `examples/chat/`              | Multi-turn Conversation API with JSON messages                     |
| `examples/conversation/`      | System prompt + tools + structured tool_calls                      |
| `examples/gpu/`               | GPU-backed generation + BenchmarkInfo metrics                      |
| `examples/tokenize/`          | `Engine.Tokenize` / `Detokenize` round-trip + start/stop tokens    |
| `examples/prefill-decode/`    | Explicit two-phase generation: `RunPrefill` → `RunDecode`          |
| `examples/score/`             | Candidate scoring with `ScoreTexts` + `Score` / `TokenLength`      |
| `examples/cancel/`            | Cancelling an in-flight streaming generation                       |



## License

Apache-2.0, same as LiteRT-LM itself.
