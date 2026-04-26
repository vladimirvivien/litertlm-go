# Building the LiteRT-LM C Shared Libraries on Linux and macOS

`litertlm-go` is a Go wrapper API that loads LiteRT-LM shared libraries
(`*.so` on Linux, `*.dylib` on macOS) at runtime. Currently, the project is
not distributed with pre-built shared libraries that expose a C API. So,
you must build them yourself.

This guide walks you through the steps to build the libraries from source
so you can use them in your Go programs on **Linux and macOS**.

> **Building on Windows?** The Windows toolchain (MSVC) needs additional
> linker options and runtime staging steps that this guide does not cover.
> See [`LITERTLM-BUILD-WINDOWS.md`](./LITERTLM-BUILD-WINDOWS.md) instead.

Find more information about building (C++) [here](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/getting-started/build-and-run.md).

## Prerequisites

Install your Linux/MacOS prerequisites per [LiteRT-LM's official 
guide](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/getting-started/build-and-run.md):

| Platform | Install |
|----------|---------|
| Linux    | `sudo apt install clang git-lfs` |
| macOS    | `xcode-select --install` and `brew install git-lfs` |

## Build via Docker (optional)

For convenience, this repo ships a `Dockerfile` that downloads the required tools
to build C library files automatically. The following snippet shows the build steps
using Docker.

```bash
export LITERTLM_LIB=~/include/litertlm/lib
mkdir -p $LITERTLM_LIB

# CPU-only
docker build --target cpu -o $LITERTLM_LIB .

# GPU-capable (still needs host Vulkan drivers at runtime)
docker build --target gpu -o $LITERTLM_LIB .
```

`$LITERTLM_LIB` ends up with `liblitertlm_c[_cpu].so` alongside the prebuilt
runtime dependencies.

## Build from source
If you want to build manually, instead of using Docker, follow the instructions in this section.

### 1. Clone the LiteRT-LM repo
```bash
git clone https://github.com/google-ai-edge/LiteRT-LM.git
cd LiteRT-LM
git lfs install --local
git lfs pull
```
Confirm the prebuilt dependencies for your target are present:

```bash
ls prebuilt/
```
### 2. Create a Bazel BUILD file

Create Bazel file `c/litertlm_c_api/BUILD` with bulid
targets to create the C API shared libraries:

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
### 3. Build the shared libraries
Run `bazel` to build the binaries. The GPU-capable target needs two extra `--define` flags.

```bash
# CPU-only
bazel build //c/litertlm_c_api:litertlm_c_cpu

# GPU-capable
bazel build //c/litertlm_c_api:litertlm_c \
    --define=litert_link_capi_so=true \
    --define=resolve_symbols_in_exec=false
```

By default, the built files are stored at `bazel-bin/c/litertlm_c_api/*.so` on Linux
and `*.dylib` on macOS.

Use `bazel clean --expunge` if you need to clear the build and start over.

### 4. Stage the libraries

You will need to store all library files in their dependencies in a known location.
Store that location in environment variable `LITERTLM_LIB` for easy reference:

```bash
export LITERTLM_LIB=~/include/litertlm/lib
mkdir -p $LITERTLM_LIB
```

```bash
# Linux
cp prebuilt/linux_x86_64/*.so $LITERTLM_LIB              # all runtime deps
cp bazel-bin/c/litertlm_c_api/liblitertlm_c*.so $LITERTLM_LIB  # ONLY your build

# macOS
cp prebuilt/macos_arm64/*.dylib $LITERTLM_LIB
cp bazel-bin/c/litertlm_c_api/liblitertlm_c*.dylib $LITERTLM_LIB
```

## Get a model
Next, you will need to download an LLM. This document 
uses Gemma 4 model prepared for LiteRT-LM.

You can download a `.litertlm` model from the
[LiteRT Community](https://huggingface.co/litert-community) on Hugging Face.
The examples below use `gemma-4-E2B-it.litertlm`.

## Run an examples

Assuming `LITERTLM_LIB` points to the location of your shared libraries, you can
test your setup with the following example:

```bash
LITERTLM_LIB=~/include/litertlm/lib go run ./examples/hello \
    -model ~/models/gemma-4-E2B-it.litertlm \
    -backend cpu
```

## Troubleshooting


| Symptom | Cause | Fix |
|---------|-------|-----|
| Link error *"not an object or archive"* on a `prebuilt/*.so` | LFS pointer, not the binary | `git lfs install --local && git lfs pull` |
| Bazel: *"Cannot find gcc or CC (clang)"* (Linux) | clang missing | `sudo apt install clang` |
| `clang: error: unknown argument: '-mavxvnniint8'` | clang ≤ 14; XNNPACK needs clang ≥ 16 | Install `clang-16+` (Ubuntu 22.04 / Debian 12 default `clang` is too old) |
| Bazel: *"requires Bazel 7.6.1 …"* | Wrong Bazel on `PATH` | Use Bazelisk |
| Runtime: *"error while loading shared libraries: libLiteRt.so"* | GPU plugins not in `LITERTLM_LIB` | Re-run §4 |
| `engine_create` returns NULL early in setup | LFS deps missing or stale | `git lfs pull`; check file sizes |
| `engine_create` returns NULL with `DYNAMIC_UPDATE_SLICE` in logs | `max_num_tokens` below the model's smallest prefill signature (often 128) | Raise `max_num_tokens` to ≥1024 |
| `NumCandidates() == 1` but `Text(0) == ""` | Chat-tuned model got raw text without its template | Use the `chat` example / Conversation API |
| `backend=gpu` fails with *"No adapters found"* / Vulkan errors | Host has no Vulkan-capable GPU driver | Install Vulkan drivers for your GPU, or run `-backend cpu` |
