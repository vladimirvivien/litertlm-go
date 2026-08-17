# litertlm-go

A Go binding for Google's
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) for high-performance local on-device LLM inference.

> Inspired by Hybridgroup's [Yzma](https://github.com/hybridgroup/yzma).

📖 **Full documentation:**
[vladimirvivien.github.io/litertlm-go](https://vladimirvivien.github.io/litertlm-go/)

## 🔥 Features

* 📦 **Automated Provisioning** — Automatically fetch, cache, and stage official LiteRT-LM shared libraries (`LibFetch`) and `.litertlm` models from Hugging Face (`FetchModel`) in Go.
* ⚙️ **Centralized Configuration** — Configure backends, token limits, and sampler parameters from a shared `config.json` file (`WithConfigFile`).
* 💬 **[Stateful Chat & Conversations](chat.md)** — Multi-turn chat orchestration with system prompts, transcript seeding, KV cache management, and conversation branching (`Chat.Clone`).
* 🖼️ **[Multimodal Inputs](client.md#multimodal-inputs)** — Process text, images, and audio inputs in any order using a unified Go interface.
* 🛠️ **[Automated Tool Calling](tools.md)** — Register standard Go functions as tools with automatic dispatch, streaming tool calls, and error recovery policies.
* 🎯 **[Structured JSON Output](structured-output.md)** — Extract model outputs directly into Go structs with type-safe generic helpers.
* ⚡ **High Performance & GPU Acceleration** — Run on CPU, GPU (WebGPU/Direct3D 12/Metal), or NPU with multi-token speculative decoding.
* 🔍 **[Low-Level Control](low-level.md)** — Access the full LiteRT-LM C-API when you need custom inference loops, scoring, template introspection, or file-descriptor loading.
* 🤖 **[Supported Models](models.md)** — Compatible with Gemma 4, Gemma 3/3n, Qwen 3/2.5, Phi-4, and other LiteRT-LM model families.

## Install

```bash
go get github.com/vladimirvivien/litertlm-go@latest
```

## Quickstart (with Automated Provisioning)

Here is how simple it is to get started with `litertlm-go`. Start by programmatically download LiteRT-LM libraries and model artifacts directly in your Go application:


```go
package main

import (
	"context"
	"fmt"
	"log"
	"runtime"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	ctx := context.Background()

	// 1. Automatically fetch and cache native libraries
	libDir, err := litertlm.LibFetch(runtime.GOOS, runtime.GOARCH, "v0.16.0")
	if err != nil {
		log.Fatalf("LibFetch failed: %v", err)
	}

	// 2. Automatically download model from Hugging Face
	modelPath, err := litertlm.FetchModel(ctx, "litert-community/gemma3-1b-it-int4")
	if err != nil {
		log.Fatalf("FetchModel failed: %v", err)
	}

	// 3. Initialize client and generate text
	client, err := litertlm.New(ctx,
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
	)
	if err != nil {
		log.Fatalf("Initialization failed: %v", err)
	}
	defer client.Close()

	text, err := client.Generate(ctx, "Write a haiku about the sea.")
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}
	fmt.Println(text)
}
```

### Manual Provisioning

You can optionally run with pre-existing local libraries and model paths:

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
LITERTLM_MODEL=/abs/path/to/gemma3-1b-it-int4.litertlm \
    go run main.go
```

Full walkthrough: [Getting started](getting-started.md).

## Provisioning the C libraries

LiteRT-LM publishes prebuilt C-API shared libraries (`v0.16.0+`) for Linux, macOS, and Windows. Use `litertlm.LibFetch` in Go to programmatically stage dependencies, or consult the platform setup guides:
[`LITERTLM-BUILD.md`](https://github.com/vladimirvivien/litertlm-go/blob/main/LITERTLM-BUILD.md)
(Linux/macOS) or
[`LITERTLM-BUILD-WINDOWS.md`](https://github.com/vladimirvivien/litertlm-go/blob/main/LITERTLM-BUILD-WINDOWS.md) (Windows).

Minimum supported upstream: **LiteRT-LM v0.16.0**.

## Examples

Self-contained programs covering every public API surface live under
[`examples/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples).
See the
[examples index](https://github.com/vladimirvivien/litertlm-go/blob/main/examples/README.md)
for the full list.

## License

Apache-2.0, same as LiteRT-LM itself.
