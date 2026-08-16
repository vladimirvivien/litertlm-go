# Getting Started with litertlm-go

`litertlm-go` supports two setup workflows:
1. **Fully Programmatic Setup (Recommended):** Automatically downloads and caches both the native C libraries and the model artifact in Go without manual setup steps.
2. **Manual Provisioning:** Manually download the `.litertlm` model file and native shared libraries, specifying their paths via environment variables or options.

---

## 1. Prerequisites

1. **Go 1.26 or newer**.
2. **Install the package:**
   ```bash
   go get github.com/vladimirvivien/litertlm-go@latest
   ```

---

## 2. Method 1: Fully Programmatic Setup (Recommended)

In this workflow, Go automatically fetches the platform prebuilt libraries (`litertlm.LibFetch`) and downloads the `.litertlm` model from Hugging Face (`litertlm.FetchModel`):

### Go Program (`main.go`)

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

	// 1. Automatically fetch and cache native LiteRT-LM shared libraries
	libDir, err := litertlm.LibFetch(runtime.GOOS, runtime.GOARCH, "v0.16.0")
	if err != nil {
		log.Fatalf("LibFetch failed: %v", err)
	}

	// 2. Automatically download and cache the model from Hugging Face
	modelPath, err := litertlm.FetchModel(ctx, "litert-community/gemma3-1b-it-int4")
	if err != nil {
		log.Fatalf("FetchModel failed: %v", err)
	}

	// 3. Initialize the client using the staged library and model
	client, err := litertlm.New(ctx,
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
	)
	if err != nil {
		log.Fatalf("Client initialization failed: %v", err)
	}
	defer client.Close()

	// 4. Generate text
	text, err := client.Generate(ctx, "Write a haiku about the sea.")
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}
	fmt.Println(text)
}
```

### Running the Program

```bash
go run main.go
```

---

## 3. Method 2: Manual Model & Library Provisioning

If you already have a model file and native shared libraries on disk:

### Step 1: Obtain the Model File

Download a `.litertlm` model from Hugging Face's [LiteRT Community](https://huggingface.co/litert-community) (e.g., `gemma3-1b-it-int4.litertlm` or `gemma-4-E2B-it.litertlm`).

### Step 2: Obtain the Shared Libraries

Download the prebuilt release archive for your platform from the [LiteRT-LM Releases](https://github.com/google-ai-edge/LiteRT-LM/releases/tag/v0.16.0):

* **Linux x86_64:**
  ```bash
  mkdir -p ~/include/litertlm/lib
  curl -LO https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.16.0/litertlm-linux-x86_64-v0.16.0.tar.gz
  tar -xzf litertlm-linux-x86_64-v0.16.0.tar.gz -C ~/include/litertlm/lib
  ```

* **macOS ARM64 (Apple Silicon):**
  ```bash
  mkdir -p ~/include/litertlm/lib
  curl -LO https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.16.0/CLiteRTLM_mac.xcframework.zip
  unzip CLiteRTLM_mac.xcframework.zip -d ~/include/litertlm/lib
  ```

* **Windows x86_64:**
  ```powershell
  $Env:LITERTLM_LIB = "$Env:USERPROFILE\include\litertlm\lib"
  New-Item -ItemType Directory -Path $Env:LITERTLM_LIB -Force | Out-Null
  Invoke-WebRequest -Uri https://github.com/google-ai-edge/LiteRT-LM/releases/download/v0.16.0/litertlm-windows-x86_64-v0.16.0.zip -OutFile litertlm.zip
  Expand-Archive -Path litertlm.zip -DestinationPath $Env:LITERTLM_LIB -Force
  ```

### Step 3: Go Program (`main.go`)

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
		fmt.Printf("Initialization failed: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	text, err := client.Generate(ctx, "Write a haiku about the sea.")
	if err != nil {
		fmt.Printf("Generation failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(text)
}
```

### Step 4: Run with Environment Variables

```bash
LITERTLM_LIB=~/include/litertlm/lib \
LITERTLM_MODEL=~/models/gemma3-1b-it-int4.litertlm \
    go run main.go
```

On Windows (PowerShell):
```powershell
$Env:LITERTLM_LIB = "$Env:USERPROFILE\include\litertlm\lib"
$Env:LITERTLM_MODEL = "C:\models\gemma3-1b-it-int4.litertlm"
go run main.go
```

---

## 4. Expected Output

Both methods produce text completions directly from the model:

```text
Blue whispers rise and fall,
Waves crash and secrets hold low,
Ocean breathes its peace now.
```

---

## Next Steps

* **Multi-turn chat & conversations** → [Chat Guide](chat.md)
* **Automated tool and function calling** → [Tools Guide](tools.md)
* **Structured JSON decoding** → [Structured Output Guide](structured-output.md)
* **Model scaling and memory sizing** → [Supported Models](models.md)
* **Low-level C-API control** → [Low-Level API](low-level.md)
