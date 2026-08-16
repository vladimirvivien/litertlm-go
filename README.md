# litertlm-go

[![Go Reference](https://pkg.go.dev/badge/github.com/vladimirvivien/litertlm-go.svg)](https://pkg.go.dev/github.com/vladimirvivien/litertlm-go)

A high-performance, cgo-free, purego-backed Go binding for Google's [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime, designed for local on-device LLM inference.

> [!NOTE]
> Inspired by Hybridgroup's [Yzma](https://github.com/hybridgroup/yzma).

📖 **Full documentation:** [vladimirvivien.github.io/litertlm-go](https://vladimirvivien.github.io/litertlm-go/)

---

## 🔥 Features

* 💬 **Stateful Chat & Conversations** — Multi-turn chat orchestration with system prompts, message history, and token tracking.
* 🖼️ **Multimodal Inputs** — Process text, images, and audio inputs in any order using a unified Go interface.
* 🛠️ **Automated Tool Calling** — Register standard Go functions as tools that the model can call automatically and in parallel.
* 🎯 **Structured JSON Output** — Extract model outputs directly into Go structs with type-safe generic helpers.
* ⚡ **High Performance & GPU Acceleration** — Run on CPU or GPU (WebGPU/Direct3D 12) with built-in memory safety.
* 🔍 **Low-Level Control** — Access the full LiteRT-LM C-API when you need custom inference loops, scoring, or custom memory management.

---

## ⚙️ Install

```bash
go get github.com/vladimirvivien/litertlm-go@latest
```

---

## 🚀 Quickstart

### Programmatic Setup (Automated)

Download both native prebuilts and model artifacts on-the-fly in Go:

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

	// 1. Fetch prebuilt libraries automatically
	libDir, err := litertlm.LibFetch(runtime.GOOS, runtime.GOARCH, "v0.16.0")
	if err != nil {
		log.Fatalf("LibFetch failed: %v", err)
	}

	// 2. Download model from Hugging Face automatically
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

Run with pre-existing local libraries and model paths:

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
LITERTLM_MODEL=/abs/path/to/gemma3-1b-it-int4.litertlm \
    go run main.go
```

> [!TIP]
> To run on the GPU backend, configure initialization with `litertlm.WithBackend("gpu")` (defaults to `"cpu"`).

---

## 📦 Provisioning & Staging

* **Automated Provisioning (Go):** Use `litertlm.LibFetch` and `litertlm.FetchModel`.
* **Linux / macOS Manual Staging:** [`LITERTLM-BUILD.md`](./LITERTLM-BUILD.md)
* **Windows Manual Staging & DXC Setup:** [`LITERTLM-BUILD-WINDOWS.md`](./LITERTLM-BUILD-WINDOWS.md)

> [!NOTE]
> For developers modifying the C++ engine itself, both build guides also include instructions for building from source using Bazel. Minimum supported upstream: **LiteRT-LM v0.16.0**.

---

## 🤖 Examples

The repository includes a variety of examples showcasing advanced features. For example, [./examples/bot](./examples/bot) demonstrates the OCR capabilities of the Gemma 4 family, extracting structured information from an image of a form:

![OCR example](./docs/litertlm-ocr-2-short.gif)

See [`examples/README.md`](./examples/README.md) for a full list of available examples.

---

## 📖 Documentation

* [Getting Started](https://vladimirvivien.github.io/litertlm-go/getting-started/) — Installation and library setup
* [Client Guide](https://vladimirvivien.github.io/litertlm-go/client/) — `New`, `Generate`, and unary/multimodal configurations
* [Chat Guide](https://vladimirvivien.github.io/litertlm-go/chat/) — Multi-turn history, system prompts, and tool calling
* [Structured Output](https://vladimirvivien.github.io/litertlm-go/structured-output/) — JSON extraction via `GenerateData[T]`
* [Low-Level API](https://vladimirvivien.github.io/litertlm-go/low-level/) — Custom inference, scoring, and memory management
* [Supported Models](https://vladimirvivien.github.io/litertlm-go/docs/models.md) — Supported model families and template processor mappings
* [Troubleshooting](https://vladimirvivien.github.io/litertlm-go/troubleshooting/) — Debugging load, memory, and acceleration issues

---

## 📄 License

Apache-2.0, same as LiteRT-LM itself.
