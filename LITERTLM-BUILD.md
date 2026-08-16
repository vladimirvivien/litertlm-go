# Staging & Building LiteRT-LM C Shared Libraries (Linux & macOS)

`litertlm-go` loads LiteRT-LM shared libraries (`*.so` on Linux, `*.dylib` on macOS) at runtime. Starting with upstream **v0.16.0**, official prebuilt shared libraries are distributed for Linux and macOS.

---

## 1. Staging Prebuilt Libraries (Recommended)

### Option A: Automated Provisioning via `litertlm.LibFetch`

`litertlm-go` includes a built-in helper to automatically download and stage the official prebuilts:

```go
package main

import (
    "fmt"
    "log"
    "runtime"

    "github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
    libDir, err := litertlm.LibFetch(runtime.GOOS, runtime.GOARCH, "v0.16.0")
    if err != nil {
        log.Fatalf("LibFetch failed: %v", err)
    }
    fmt.Printf("LiteRT-LM libraries staged at: %s\n", libDir)
}
```

### Option B: Manual Download

1. Download the prebuilt release archive for your platform from the [LiteRT-LM Releases](https://github.com/google-ai-edge/LiteRT-LM/releases/tag/v0.16.0):
   * **Linux x86_64:** `litertlm-linux-x86_64-v0.16.0.tar.gz`
   * **Linux ARM64:** `litertlm-linux-arm64-v0.16.0.tar.gz`
   * **macOS ARM64 (Apple Silicon):** `litertlm-macos-arm64-v0.16.0.tar.gz`
2. Extract the archive to a local staging directory (e.g., `~/include/litertlm/lib`):
   ```bash
   mkdir -p ~/include/litertlm/lib
   tar -xzf litertlm-linux-x86_64-v0.16.0.tar.gz -C ~/include/litertlm/lib
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
git checkout v0.16.0
git lfs install --local
git lfs pull
```

### Step 2: Configure Bazel C-API Target

Create `c/litertlm_c_api/BUILD`:

```python
package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "litertlm_c_cpu",
    linkshared = 1,
    deps = ["//c:engine_cpu"],
)

cc_binary(
    name = "litertlm_c",
    linkshared = 1,
    deps = ["//c:engine"],
)
```

### Step 3: Build Shared Libraries

```bash
# CPU library
bazelisk build //c/litertlm_c_api:litertlm_c_cpu

# GPU library (requires dynamic linking flags)
bazelisk build //c/litertlm_c_api:litertlm_c \
    --define=litert_runtime_link_mode=dynamic \
    --define=resolve_symbols_in_exec=false
```

### Step 4: Stage Built Binaries

Copy the compiled libraries and upstream plugin dependencies to your staging directory:

```bash
export LITERTLM_LIB=~/include/litertlm/lib
mkdir -p $LITERTLM_LIB

# Linux
cp prebuilt/linux_x86_64/*.so $LITERTLM_LIB
cp bazel-bin/c/litertlm_c_api/liblitertlm_c*.so $LITERTLM_LIB

# macOS
cp prebuilt/macos_arm64/*.dylib $LITERTLM_LIB
cp bazel-bin/c/litertlm_c_api/liblitertlm_c*.dylib $LITERTLM_LIB
```

---

## 4. Troubleshooting

| Symptom | Cause | Solution |
|---|---|---|
| Link error *"not an object or archive"* on a `prebuilt/*.so` | LFS pointer file, not binary | Run `git lfs install --local && git lfs pull` |
| `cannot open shared object file: libLiteRt.so` | Auxiliary GPU dependencies missing from directory | Ensure `prebuilt/<platform>/*` files are copied into `$LITERTLM_LIB` |
| `engine_create` returns `NULL` | Model path invalid or insufficient memory | Check model file integrity and verify system RAM/VRAM |
| Empty `default.profraw` files appear on exit | Upstream binaries built with LLVM profiling | Set `export LLVM_PROFILE_FILE=/dev/null` in environment |
