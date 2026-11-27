# litertlm-go

A Go binding for Google's
[LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) for high-performance local on-device LLM inference.

> Inspired by Hybridgroup's [Yzma](https://github.com/hybridgroup/yzma).

📖 **Full documentation:**
[vladimirvivien.github.io/litertlm-go](https://vladimirvivien.github.io/litertlm-go/)

## 🔥 Features

* 📦 **Automated Provisioning & Differential Sync** — Fetch, cache, and stage official LiteRT-LM shared libraries (`FetchLib`), download `.litertlm` models from Hugging Face (`FetchModel`), and reuse unchanged sections during model updates (`PlanModelSync`, `SyncModel`, `SyncModelCache`, C API `1.1+`).
* 🔬 **Model Inspection & Structured Diagnostics** — Inspect model types, context limits, hardware backends/NPU targets, thinking support, function calling, embedding dimensions, and default sampler settings without booting an engine (`InspectModel`), with `*CError` diagnostics and `StatusCode` sentinel matching.
* ⚙️ **Centralized Configuration** — Configure backends, speculative decoding, token limits, and sampler parameters from a shared `config.json` file (`WithConfigFile`).
* 💬 **[Stateful Chat & Conversations](chat.md)** — Multi-turn chat orchestration with system prompts, custom prompt templates, thinking budgets, transcript seeding, KV-cache channel filtering, and branching (`Chat.Clone`).
* 🎯 **Constrained Decoding & Logit Controls** — Enforce JSON Schema or regex grammars (`WithConstraint`), repetition/presence/frequency penalties, n-gram blocking, and token suppression.
* 🖼️ **[Multimodal Inputs](client.md#multimodal-inputs)** — Process text, images, and audio inputs in any order using a unified Go interface.
* 🧭 **[Standalone Embeddings](embeddings.md)** — Generate text, multimodal, batch, and Matryoshka-truncated embeddings with `EmbeddingClient`.
* 🛠️ **[Automated Tool Calling](tools.md)** — Register Go functions as tools with automatic dispatch, streaming tool calls, error recovery policies, and `ToolChoice` selection policies (C API `1.1+`).
* 📐 **[Structured JSON Output](structured-output.md)** — Extract model outputs directly into Go structs with type-safe generic helpers (`GenerateData[T]`, `GenerateDataMulti[T]`).
* ⚡ **Hardware Acceleration, Residency & Diagnostics** — Run on CPU (XNNPACK / YNNPACK), GPU (WebGPU / Direct3D 12 / Metal), or NPU with configurable speculative decoding (`Off`, `On`, `Adaptive`), runtime Metal residency toggles (`Client.SetMetalResidency`), and AI Edge Debugger capture inspection (`WithDebuggerEnabled`, C API `1.1+`).
* 🔍 **[Low-Level Control](low-level.md)** — Access the LiteRT-LM C API for explicit prefill/decode, KV-cache checkpoints and rewinding, session cloning, async decode, scoring, tokenization, and file-descriptor loading.
* 🤖 **[Supported Models](models.md)** — Compatible with Gemma 4, Gemma 3/3n, Qwen 3/2.5, Phi-4, and other LiteRT-LM model families.

## Install

```bash
go get github.com/vladimirvivien/litertlm-go@latest
```

## Quickstart

### Automated Provisioning (Recommended)

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
	libDir, err := litertlm.FetchLib(ctx, runtime.GOOS, runtime.GOARCH, "v0.19.1")
	if err != nil {
		log.Fatalf("FetchLib failed: %v", err)
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

### Manual Setup

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
LITERTLM_MODEL=/abs/path/to/gemma3-1b-it-int4.litertlm \
    go run main.go
```

Full walkthrough: [Getting started](getting-started.md).

## Provisioning the C libraries

LiteRT-LM publishes prebuilt C-API shared libraries (`v0.19.1+`, C API `1.0.0+`) for Linux, macOS, and Windows. Use `litertlm.FetchLib` in Go to programmatically stage dependencies, or consult the platform setup guides:
[`LITERTLM-BUILD.md`](https://github.com/vladimirvivien/litertlm-go/blob/main/LITERTLM-BUILD.md)
(Linux/macOS) or
[`LITERTLM-BUILD-WINDOWS.md`](https://github.com/vladimirvivien/litertlm-go/blob/main/LITERTLM-BUILD-WINDOWS.md) (Windows).

Minimum supported upstream: **LiteRT-LM v0.19.1** (C API `1.0.0`).

## Examples

48 self-contained runnable programs covering every public API surface live under
[`examples/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples).
See the
[examples index](https://github.com/vladimirvivien/litertlm-go/blob/main/examples/README.md)
for the full list.

## License

Apache-2.0, same as LiteRT-LM itself.
