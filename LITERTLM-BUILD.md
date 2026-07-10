# Building the LiteRT-LM C Shared Libraries on Linux and macOS

`litertlm-go` is a Go wrapper API that loads LiteRT-LM shared libraries
(`*.so` on Linux, `*.dylib` on macOS) at runtime. Currently, LiteRT-LM is
not distributed with pre-built shared libraries. So,
you must build them yourself.

> **Building on Windows?** The Windows toolchain (MSVC) needs additional
> linker options and runtime staging steps that this guide does not cover.
> See [`LITERTLM-BUILD-WINDOWS.md`](./LITERTLM-BUILD-WINDOWS.md) instead.

## Prerequisites

First, install your Linux/MacOS prerequisites per [LiteRT-LM's official 
guide](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/getting-started/build-and-run.md):

| Platform | Install |
|----------|---------|
| Linux    | `sudo apt install clang git-lfs` |
| macOS    | `xcode-select --install` and `brew install git-lfs` |

## Build via Docker

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

`$LITERTLM_LIB` should contain `liblitertlm_c[_cpu].so` alongside the prebuilt
runtime dependencies.

## Build from source
If you want to build manually, follow the instructions in this section.

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

```bash
# CPU
bazel build //c/litertlm_c_api:litertlm_c_cpu

# GPU
bazel build //c/litertlm_c_api:litertlm_c \
    --define=litert_runtime_link_mode=dynamic \
    --define=resolve_symbols_in_exec=false
```

The two `--define` flags are required for the GPU target. See [Build
mechanics](#build-mechanics) for what they do.

Built files land in `bazel-bin/c/litertlm_c_api/*.so` on Linux and `*.dylib` on
macOS. Run `bazel clean --expunge` to start over from scratch.

### 3.5 Verify the build

The GPU shared library must dynamically link `libLiteRt`:

```bash
# Linux
objdump -p bazel-bin/c/litertlm_c_api/liblitertlm_c.so | grep NEEDED
# must include: NEEDED  libLiteRt.so

# macOS
otool -L bazel-bin/c/litertlm_c_api/liblitertlm_c.dylib | grep -i litert
# must list libLiteRt.dylib
```

If `libLiteRt` is absent from the output, rebuild per §3 with both `--define`
flags.

### 4. Stage the libraries

Next, store all library files in a known location (`LITERTLM_LIB`) to make them easy to find:

```bash
export LITERTLM_LIB=~/include/litertlm/lib
mkdir -p $LITERTLM_LIB
```

```bash
# Linux
cp prebuilt/linux_x86_64/*.so $LITERTLM_LIB                    # all runtime deps
cp bazel-bin/c/litertlm_c_api/liblitertlm_c*.so $LITERTLM_LIB  # your C API build

# macOS
cp prebuilt/macos_arm64/*.dylib $LITERTLM_LIB
cp bazel-bin/c/litertlm_c_api/liblitertlm_c*.dylib $LITERTLM_LIB
```

## Get a model
Next, you will need to download an LLM. This document 
uses the Gemma 4 model from Hugging Face.

You can download the Gemma 4 `.litertlm` model from the
[LiteRT Community](https://huggingface.co/litert-community) on Hugging Face.

## Test your build

Assuming `LITERTLM_LIB` points to the location of your shared libraries, you can
test your setup with the following examples:

CPU inference:

```bash
LITERTLM_LIB=~/include/litertlm/lib go run ./examples/hello \
    -model ~/models/gemma-4-E2B-it.litertlm \
    -backend cpu
```

GPU-backed inference (Linux):

```bash
LITERTLM_LIB=~/include/litertlm/lib \
LD_LIBRARY_PATH=$LITERTLM_LIB:$LD_LIBRARY_PATH \
LLVM_PROFILE_FILE=/dev/null \
go run ./examples/chat -model ~/models/gemma-4-E2B-it.litertlm -backend gpu
```

GPU-backed inference (macOS):

```bash
LITERTLM_LIB=~/include/litertlm/lib \
DYLD_LIBRARY_PATH=$LITERTLM_LIB:$DYLD_LIBRARY_PATH \
LLVM_PROFILE_FILE=/dev/null \
go run ./examples/chat -model ~/models/gemma-4-E2B-it.litertlm -backend gpu
```

## Build mechanics

`liblitertlm_c.{so,dylib}` is a thin shim around upstream's `c/engine.{cc,h}`.
It must share the LiteRT runtime instance with the GPU accelerator plugin
(`libLiteRtWebGpuAccelerator.{so,dylib}`), which loads `libLiteRt.{so,dylib}` at
runtime.

Two `--define` flags control how the runtime is linked into the C-API
shared library:

- `--define=litert_runtime_link_mode=dynamic` — links the C-API shared library against
  `libLiteRt.{so,dylib}` dynamically. The C-API library and the accelerator
  plugin then share one TFLite delegate registry through the same
  `libLiteRt.{so,dylib}`. Required for GPU.
- `--define=resolve_symbols_in_exec=false` — required companion when linking
  dynamically.

The CPU target (`litertlm_c_cpu`) uses `//runtime/core:engine_impl_cpu_only`,
which has no accelerator-plugin interaction. The flags are optional for CPU.

The Python (`python/litert_lm/BUILD`) and Kotlin
(`kotlin/java/com/google/ai/edge/litertlm/jni/BUILD`) bindings follow the same
convention: the BUILD file is link-mode-agnostic, and the `--define` flag at
the command line controls static vs dynamic. Upstream CI passes both flags on
every platform (`.github/workflows/ci-build.yml`).

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Link error *"not an object or archive"* on a `prebuilt/*.so` | LFS pointer, not the binary | `git lfs install --local && git lfs pull` |
| Bazel: *"Cannot find gcc or CC (clang)"* (Linux) | clang missing | `sudo apt install clang` |
| `clang: error: unknown argument: '-mavxvnniint8'` | clang ≤ 14; XNNPACK needs clang ≥ 16 | Install `clang-16+` (Ubuntu 22.04 / Debian 12 default `clang` is too old) |
| Bazel: *"requires Bazel 7.6.1 …"* | Wrong Bazel on `PATH` | Use Bazelisk |
| Runtime: *"error while loading shared libraries: libLiteRt.so"* | GPU plugins not in `LITERTLM_LIB` | Re-run §4 |
| `litertlm: New: ...liblitertlm_c_cpu.{so,dylib}: cannot open shared object file` | Only the GPU/full build was staged; `-backend cpu` looks for `liblitertlm_c_cpu.*` | Either build `litertlm_c_cpu` (§3) and copy it in, or symlink the GPU build under the CPU name (see §4 *One staging dir for both backends*) |
| `engine_create` returns NULL early in setup | LFS deps missing or stale | `git lfs pull`; check file sizes |
| `engine_create` returns NULL with `DYNAMIC_UPDATE_SLICE` in logs | `max_num_tokens` below the model's smallest prefill signature (often 128) | Raise `max_num_tokens` to ≥1024 |
| `NumCandidates() == 1` but `Text(0) == ""` | Chat-tuned model got raw text without its template | Use the `chat` example / Conversation API |
| `backend=gpu` fails with *"No adapters found"* / Vulkan errors | Host has no Vulkan-capable GPU driver | Install Vulkan drivers for your GPU, or run `-backend cpu` |
| Empty `default.profraw` files appear in your working directory after each run | The prebuilt LiteRT-LM deps (`prebuilt/<os>/lib*.so`) were compiled with LLVM `-fprofile-instr-generate`; the embedded `__llvm_profile_*` runtime writes a coverage dump to `./default.profraw` on exit | Set `LLVM_PROFILE_FILE=/dev/null` (or any throw-away path) in the environment to discard the dump: `LLVM_PROFILE_FILE=/dev/null LITERTLM_LIB=… go run …` |
