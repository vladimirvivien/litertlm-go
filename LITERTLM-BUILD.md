# Staging & Building LiteRT-LM C Shared Libraries (Linux & macOS)

`litertlm-go` loads the unified `liblitert-lm` shared library (`liblitert-lm.so` on Linux, `liblitert-lm.dylib` on macOS) targeting LiteRT-LM **v0.19.1** (C API `1.0.0`). Official prebuilt shared libraries are distributed for Linux and macOS.

---

## 1. Staging Prebuilt Libraries (Recommended)

### Option A: Automated Provisioning via `litertlm.FetchLib`

`litertlm-go` includes a built-in helper to automatically download and stage the official prebuilts:

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
    libDir, err := litertlm.FetchLib(context.Background(), runtime.GOOS, runtime.GOARCH, "v0.19.1")
    if err != nil {
        log.Fatalf("FetchLib failed: %v", err)
    }
    fmt.Printf("LiteRT-LM libraries staged at: %s\n", libDir)
}
```

### Option B: Manual Download

1. Download the C-API release archive (`litert_lm_c_api-1.0.0.zip`) or macOS archive from [LiteRT-LM Releases](https://github.com/google-ai-edge/LiteRT-LM/releases/tag/v0.19.1).
2. Extract the platform shared library (`liblitert-lm.so` for Linux, `liblitert-lm.dylib` for macOS) into a local staging directory (e.g., `~/include/litertlm/lib`):
   ```bash
   mkdir -p ~/include/litertlm/lib
   unzip litert_lm_c_api-1.0.0.zip -d /tmp/litert_lm_c_api
   # Linux x86_64:
   cp /tmp/litert_lm_c_api/lib/linux_x86_64/liblitert-lm.so ~/include/litertlm/lib/
   # macOS ARM64:
   cp /tmp/litert_lm_c_api/lib/macos_arm64/liblitert-lm.dylib ~/include/litertlm/lib/
   ```
3. Set the environment variable:
   ```bash
   export LITERTLM_LIB=~/include/litertlm/lib
   ```

---

## 2. Testing Your Staged Libraries

Download a `.litertlm` model file (such as `gemma3-1b-it-int4.litertlm`) from Hugging Face's [LiteRT Community](https://huggingface.co/litert-community), then run an example:

```bash
# CPU inference
LITERTLM_LIB=~/include/litertlm/lib go run ./examples/hello \
    -model ~/models/gemma3-1b-it-int4.litertlm \
    -backend cpu

# GPU inference (Linux / macOS)
LITERTLM_LIB=~/include/litertlm/lib go run ./examples/chat \
    -model ~/models/gemma3-1b-it-int4.litertlm \
    -backend gpu
```

---

## 3. Building from Source with Bazel (Advanced / Custom Builds)

If you are modifying the C++ engine or building custom operator extensions, you can compile from source using Bazel.

### Prerequisites

| Platform | Dependencies |
|---|---|
| Linux | `clang` (v16+), `git-lfs`, `bazelisk` |
| macOS | Xcode Command Line Tools (`xcode-select --install`), `brew install git-lfs bazelisk` |

### Step 1: Clone Repository and Pull LFS Assets

```bash
git clone https://github.com/google-ai-edge/LiteRT-LM.git
cd LiteRT-LM
git checkout v0.19.1
git lfs install --local
git lfs pull
```

### Step 2: Build the Unified Shared Library

Upstream provides the unified C-API target `//c:litert-lm`:

```bash
bazelisk build //c:litert-lm
```

### Step 3: Stage Built Binaries

Copy the compiled library to your staging directory:

```bash
export LITERTLM_LIB=~/include/litertlm/lib
mkdir -p $LITERTLM_LIB

# Linux
cp bazel-bin/c/liblitert-lm.so $LITERTLM_LIB/

# macOS
cp bazel-bin/c/liblitert-lm.dylib $LITERTLM_LIB/
```

---

## 4. Troubleshooting

| Symptom | Cause | Solution |
|---|---|---|
| Link error *"not an object or archive"* on a `prebuilt/*` library | LFS pointer file, not binary | Run `git lfs install --local && git lfs pull` |
| `liblitert-lm.so: cannot open shared object file` | Library directory not staged or `LITERTLM_LIB` unset | Ensure `$LITERTLM_LIB` contains `liblitert-lm.so` (or pass `WithLib`) |
| `NewEngine` returns an initialization error | Model path invalid, unsupported backend, or insufficient memory | Check model file integrity, backend constraint, and available RAM/VRAM |
| Empty `default.profraw` files appear on exit | Upstream binaries built with LLVM profiling | Set `export LLVM_PROFILE_FILE=/dev/null` in environment |
