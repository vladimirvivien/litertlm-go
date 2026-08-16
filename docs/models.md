# Supported Models & Scaling Guide

`litertlm-go` supports `.litertlm` model artifacts produced for Google's LiteRT-LM runtime. This document outlines verified model families, parameter scaling requirements (from 1B up to 31B parameters), memory/VRAM allocation guidelines, and configuration options.

---

## 1. Verified Model Architectures (LiteRT-LM v0.16.0)

The test battery ([`pkg/litertlm/model_matrix_test.go`](file:///C:/Users/vladi/DEV/litertlm-go/pkg/litertlm/model_matrix_test.go)) validates chat generation, tokenization round-trips, and streaming coherence across both CPU and GPU backends.

| Model Family | Target Processors | Parameter Range | CPU Backend | GPU Backend (Metal / WebGPU / D3D12) |
|---|---|:---:|:---:|:---:|
| **Gemma 3** | `gemma3` | 1B, 4B, 12B, 27B | ✅ | ✅ |
| **Gemma 4** | `gemma4` | E2B, E4B, 12B | ✅ | ✅ |
| **Gemma 2** | `gemma2` | 2B, 9B, 27B | ✅ | ✅ |
| **Qwen 2.5 / Qwen 3** | `qwen3`, `generic` | 0.5B, 1.5B, 3B, 7B, 14B, 32B | ✅ | ✅ (Single-turn) |
| **Llama 3 / 3.1 / 3.2** | `generic` | 1B, 3B, 8B | ✅ | ✅ |
| **Phi-3.5 / Phi-4** | `generic` | 3.8B, 14B | ✅ | ✅ |
| **SmolLM / SmolLM2** | `generic` | 135M, 360M, 1.7B | ✅ | ✅ |

---

## 2. Parameter Scaling & Memory Allocation Matrix

Model memory consumption during inference consists of two primary components:
1. **Static Weights Footprint**: Base memory required to map and hold model parameters in memory.
2. **Dynamic KV-Cache Footprint**: Working memory allocated during prefill and decoding to hold key/value attention states for the active context window (`maxNumTokens`).

### Memory Requirements by Model Tier

| Tier | Example Models | Quantization | File Size | Minimum RAM (CPU) | Minimum VRAM (GPU) | Recommended Context (`maxNumTokens`) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Small / Edge (1B – 4B)** | Gemma 3 1B/4B<br>Gemma 4 E2B/E4B<br>SmolLM 1.7B | `int4`<br>`int8` | 0.8 GB – 2.5 GB<br>1.5 GB – 4.5 GB | 2 GB<br>4 GB | 2 GB – 4 GB | 2,048 – 4,096 |
| **Mid Tier (7B – 9B)** | Gemma 2 9B<br>Llama 3.1 8B<br>Mistral 7B | `int4`<br>`int8` | 4.0 GB – 5.5 GB<br>7.5 GB – 9.5 GB | 6 GB<br>10 GB | 6 GB – 10 GB | 4,096 – 8,192 |
| **Large Tier (12B – 14B)** | Gemma 3 12B<br>Qwen 2.5 14B<br>Phi-4 14B | `int4`<br>`int8` | 6.5 GB – 8.5 GB<br>12.0 GB – 15.0 GB | 10 GB<br>16 GB | 10 GB – 16 GB | 4,096 – 8,192 |
| **Scale Tier (26B – 31B)** | Gemma 2 27B<br>Gemma 3 27B<br>Qwen 2.5 32B | `int4`<br>`int8` | 14.0 GB – 18.0 GB<br>26.0 GB – 34.0 GB | 20 GB<br>36 GB | 18 GB – 36 GB | 4,096 – 16,384 |

---

## 3. KV-Cache & Context Window Sizing

The dynamic memory allocated for the Key-Value (KV) cache scales linearly with sequence length:

$$\text{Memory}_{\text{KV}} = 2 \times \text{layers} \times \text{KV-heads} \times \text{head-dimension} \times \text{bytes-per-element} \times \text{max-tokens}$$

### Context Length Recommendations

* **2,048 – 4,096 tokens:** Default for standard chat and single-document tasks. Leaves ample headroom on 8 GB / 16 GB host systems.
* **8,192 – 16,384 tokens:** Appropriate for document summarization and long multi-turn sessions. Ensure host system or GPU has at least 4 GB – 8 GB of memory headroom above base weight size.
* **Autosized Ringbuffers:** On models configured with dynamic ringbuffers in `config.json`, the KV cache expands on-demand rather than pre-allocating the entire `maxNumTokens` window upfront.

---

## 4. Configuration Tuning by Model Scale

### CPU Multi-Threading (`WithNumThreads`)
For models 7B and larger executing on the CPU backend, explicitly configure worker thread count to match physical CPU cores:

```go
client, err := litertlm.New(ctx,
    litertlm.WithModel("gemma3-12b-it-int4.litertlm"),
    litertlm.WithBackend("cpu"),
    litertlm.WithNumThreads(runtime.NumCPU()),
)
```

### Prefill Chunking (`WithPrefillChunkSize`)
When processing long prompts on 12B–31B models on CPU, prefill chunking limits peak memory usage and keeps CPU cache locality high:

```go
client, err := litertlm.New(ctx,
    litertlm.WithModel("gemma3-27b-it-int4.litertlm"),
    litertlm.WithBackend("cpu"),
    litertlm.WithPrefillChunkSize(512), // Process prompt in 512-token chunks
)
```

### Parallel File Loading (`WithParallelSectionLoading`)
For large model files (10 GB – 30 GB), enable parallel section loading to decrease model initialization time:

```go
client, err := litertlm.New(ctx,
    litertlm.WithModel("qwen2.5-32b-int4.litertlm"),
    litertlm.WithParallelSectionLoading(true),
)
```

### GPU Buffer Sizing & Allocations
On GPU backends using WebGPU / Direct3D 12, individual buffer allocations are subject to driver and API limits (typically ~2 GB per single buffer allocation). 
* Quantized `int4` formats (e.g., Gemma 3 12B/27B `int4`) partition tensor weights into sub-buffers to stay within standard GPU driver limits.
* If a model fails during initialization with `failed to prepare engine`, verify available dedicated VRAM or switch to the CPU backend (`WithBackend("cpu")`).

## 5. Automated Model Downloading (`litertlm.FetchModel`)

`litertlm-go` includes a built-in downloader to automatically download and stage `.litertlm` models from Hugging Face or direct URLs with progress reporting and range-based resumable transfers:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	ctx := context.Background()

	// Download Gemma 3 1B from Hugging Face with progress updates
	modelPath, err := litertlm.FetchModel(ctx, "litert-community/gemma3-1b-it-int4",
		litertlm.WithModelProgress(func(downloaded, total int64, pct float64) {
			if total > 0 {
				fmt.Printf("\rDownloading: %.1f%% (%.1f / %.1f MB)", pct, float64(downloaded)/1e6, float64(total)/1e6)
			}
		}),
	)
	if err != nil {
		log.Fatalf("FetchModel failed: %v", err)
	}
	fmt.Printf("\nModel cached at: %s\n", modelPath)
}
```

Supported identifier formats:
* **Hugging Face Shorthand:** `litert-community/gemma3-1b-it-int4` or `gemma3-1b-it-int4`
* **Full Repo + File:** `hf:google/gemma-3-1b-it:model.litertlm`
* **Direct HTTPS URL:** `https://huggingface.co/litert-community/gemma3-1b-it-int4/resolve/main/gemma3-1b-it-int4.litertlm`

---

## 6. Integration Test Execution

To execute the model matrix battery against local model files:

```bash
# Set paths to staging directory and models directory
export LITERTLM_TEST_LIB="/path/to/litertlm/lib"
export LITERTLM_TEST_MODELS_DIR="/path/to/models"

# Run tests on CPU backend
go test -race -v -run TestModelMatrix ./pkg/litertlm

# Run tests on GPU backend
export LITERTLM_TEST_BACKEND="gpu"
go test -race -v -run TestModelMatrix ./pkg/litertlm
```
