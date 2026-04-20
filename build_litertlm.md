# Building and Distributing the LiteRT-LM Shared Library

This guide covers building the LiteRT-LM C-API shared library
(`liblitertlm_c[_cpu].{so,dylib}` / `litertlm_c[_cpu].dll`) from source on
**Linux, macOS, and Windows**, and packaging it together with its required
runtime dependencies so a downstream app can be deployed without re-running
the build.

The output of this guide is a self-contained `dist/` directory you can copy
to any machine of the same OS/arch and consume from C, C++, or any other
language binding (e.g. the `litertlm-go` Go wrapper in this repository).

> **What you'll build:** a wrapper shared library that exposes the C API in
> `c/engine.h`. 

**What you won't build:** the lower-level LiteRT runtime and
> GPU accelerator plugins (`libLiteRt`, `libLiteRt*Accelerator`,
> `libGemmaModelConstraintProvider`) — those are pre-compiled binaries
> shipped in `prebuilt/<os>/` via Git LFS. Your wrapper dynamically loads
> them at run time, which is why the distribution step matters.

---

## 1. Install prerequisites (one-time, per machine)

| Platform   | What to install                                                                                                                |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Linux**  | `sudo apt install clang git-lfs patchelf` — clang is required (the project's `.bazelrc` pins `CC=clang` on Linux).             |
| **macOS**  | `xcode-select --install` (provides clang) and `brew install git-lfs`.                                                          |
| **Windows**| Visual Studio 2022 with the "Desktop development with C++" workload (provides MSVC), Git for Windows, Python 3.13, Git LFS, and you must enable **LongPathsEnabled** in the registry. |

### Bazel 7.6.1 (exact version, all OSes)

The repo's `.bazelversion` pins **7.6.1**. Use [Bazelisk](https://github.com/bazelbuild/bazelisk) — it reads `.bazelversion` and downloads the matching Bazel automatically:

```bash
# Linux
curl -L -o ~/.local/bin/bazel \
  https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
chmod +x ~/.local/bin/bazel
```

```bash
# macOS
brew install bazelisk
```

```powershell
# Windows
winget install --id=Bazel.Bazelisk -e
```

---

## 2. Get the source and pull the prebuilt deps

```bash
git clone https://github.com/google-ai-edge/LiteRT-LM.git
cd LiteRT-LM
git lfs install --local
git lfs pull
```


**Verify prebuilt files** 

MacOS/Linux:

```bash
file prebuilt/linux_x86_64/libGemmaModelConstraintProvider.so
```

On Windows:

```powershell
Get-Item prebuilt\windows_x86_64\libGemmaModelConstraintProvider.dll | Select-Object Length
```

---

## 3. Build the C-API shared library

Two Bazel targets are defined in `c/BUILD`:

- `//c:litertlm_c_cpu` — CPU-only. Smaller, no GPU runtime deps required.
- `//c:litertlm_c`     — GPU-capable. Wraps the same C API but its
  initialisation also dynamically loads the GPU accelerator plugins from
  `prebuilt/`.

### 3.1 Linux

```bash
# CPU-only
bazel build //c:litertlm_c_cpu
# Output: bazel-bin/c/liblitertlm_c_cpu.so

# GPU-capable
bazel build //c:litertlm_c \
    --define=litert_link_capi_so=true \
    --define=resolve_symbols_in_exec=false
# Output: bazel-bin/c/liblitertlm_c.so
```

### 3.2 macOS

```bash
# CPU-only
bazel build //c:litertlm_c_cpu
# Output: bazel-bin/c/liblitertlm_c_cpu.dylib

# GPU-capable
bazel build //c:litertlm_c \
    --define=litert_link_capi_so=true \
    --define=resolve_symbols_in_exec=false
# Output: bazel-bin/c/liblitertlm_c.dylib
```

### 3.3 Windows

Run inside a "x64 Native Tools Command Prompt for VS 2022" (or any shell
where MSVC is on PATH):

```powershell
# CPU-only
bazel build //c:litertlm_c_cpu --config=windows
# Outputs:
#   bazel-bin\c\litertlm_c_cpu.dll        ← the DLL
#   bazel-bin\c\litertlm_c_cpu.if.lib     ← the import library

# GPU-capable
bazel build //c:litertlm_c --config=windows `
    --define=litert_link_capi_so=true `
    --define=resolve_symbols_in_exec=false
# Outputs:
#   bazel-bin\c\litertlm_c.dll
#   bazel-bin\c\litertlm_c.if.lib
```

### 3.4 Clean rebuild

If a build misbehaves after a config change, wipe the cache and retry:

```bash
bazel clean --expunge
```

---

## 4. Package for distribution

The shared library Bazel just produced is **not self-contained**. Even the
CPU build needs one prebuilt runtime file at run time
(`libGemmaModelConstraintProvider`). The GPU build needs several more.

### 4.1 What each runtime dependency does

| File                                  | Required for                  | Why                                                                                            |
| ------------------------------------- | ----------------------------- | ---------------------------------------------------------------------------------------------- |
| `libGemmaModelConstraintProvider.*`   | **All builds (CPU & GPU)**    | Constrained-decoding plugin used during prefill and chat-template handling. Always loaded.     |
| `libLiteRt.*`                         | GPU build                     | The LiteRT (TFLite-Next) C runtime. Statically linked into the CPU build but dynamic for GPU.  |
| `libLiteRtWebGpuAccelerator.*`        | GPU build (all desktop OSes)  | WebGPU backend for graph execution.                                                            |
| `libLiteRtTopKWebGpuSampler.*`        | GPU build (all desktop OSes)  | TopK sampler implemented as a WebGPU compute kernel.                                           |
| `libLiteRtMetalAccelerator.dylib`     | GPU build (macOS only)        | Metal backend specific to Apple silicon.                                                       |
| `dxcompiler.dll`, `dxil.dll`          | GPU build (Windows only)      | Microsoft DirectX Shader Compiler runtime, required by the WebGPU backend on Windows.          |

### 4.2 Target distribution layout

The same shape works on every OS — only the file extensions differ:

```
dist/
├── include/
│   ├── engine.h                                ← public C API
│   └── litert_lm_logging.h                     ← optional logging API
└── lib/
    ├── liblitertlm_c[_cpu].so   |  .dylib   |  litertlm_c[_cpu].dll
    ├── libGemmaModelConstraintProvider.{so,dylib,dll}      ← always
    │
    │   --- GPU only (omit for CPU build): ---
    ├── libLiteRt.{so,dylib,dll}
    ├── libLiteRtWebGpuAccelerator.{so,dylib,dll}
    ├── libLiteRtTopKWebGpuSampler.{so,dylib,dll}
    ├── libLiteRtMetalAccelerator.dylib                     ← macOS only
    └── (Windows GPU also: dxcompiler.dll, dxil.dll, litertlm_c[_cpu].if.lib)
```

For consumers that link statically against the DLL on Windows, also include
the import library `litertlm_c[_cpu].if.lib` from `bazel-bin/c/`.

### 4.7 Download a model

LiteRT-LM consumes `.litertlm` model files (not raw `.tflite`). Pick one
from the "Supported Models and Performance" table in the upstream
LiteRT-LM README. For example, Gemma 4 instruct is published on
Hugging Face at `litert-community/gemma-4-E2B-it-litert-lm` — see the
project README for download options. Place the file anywhere; the path is
passed at runtime.

## 6. Using litertlm-go

Point the Go wrapper at the staged `dist/lib` via the `LITERTLM_LIB`
environment variable:

```bash
# Linux / macOS
LITERTLM_LIB=$(pwd)/dist/lib \
    go run ./examples/hello -model /path/to/model.litertlm

# Windows
$Env:LITERTLM_LIB = "$PWD\dist\lib"
go run .\examples\hello -model C:\path\to\model.litertlm
```

The wrapper handles the platform-specific lib filename mapping
(`liblitertlm_c_cpu.so` on Linux, `.dylib` on macOS, `litertlm_c_cpu.dll`
on Windows) and dlopens the runtime deps in the right order so neither
`LD_LIBRARY_PATH` nor `PATH` tweaks are required.

---

## 7. Troubleshooting

| Symptom                                                                | Cause                                                                                  | Fix                                                                                            |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Link error: *"not an object or archive"* on a `prebuilt/*.so`          | LFS pointer instead of binary                                                          | `git lfs install --local && git lfs pull`                                                      |
| Bazel: *"Cannot find gcc or CC (clang)"* (Linux)                       | clang not installed                                                                    | `sudo apt install clang`                                                                       |
| Bazel: *"requires Bazel 7.6.1 …"*                                      | Wrong Bazel version on PATH                                                            | Install Bazelisk (§1)                                                                          |
| Windows: file-permission errors deep in `external/`                    | LongPaths not enabled                                                                  | Set `LongPathsEnabled=1` in registry; optionally `bazelisk --output_base=C:\bzl`              |
| Runtime: *"error while loading shared libraries: libLiteRt.so"*        | GPU plugins missing from `dist/lib/`                                                   | Re-run §4.4 (Linux GPU) or §4.5 (macOS GPU) packaging                                          |
| Runtime: *"dyld: Library not loaded: bazel-out/..."* (macOS)           | Install names not rewritten                                                            | Run the `install_name_tool -id` loop in §4.5                                                   |
| Runtime: *"The code execution cannot proceed because litertlm_c_cpu.dll was not found"* (Windows) | DLL not on PATH / not next to `.exe`              | Add `dist\lib` to PATH or copy DLLs next to the executable (§5.3)                              |
| `engine_create` returns NULL early in setup                            | LFS deps missing or out of date                                                        | Re-run `git lfs pull`; verify file size/`file` output                                          |
| `engine_create` returns NULL with `DYNAMIC_UPDATE_SLICE` error in logs | `max_num_tokens` is smaller than the model's smallest prefill signature (often 128)    | Bump `max_num_tokens` to ≥1024                                                                 |
| `Responses.NumCandidates() == 1` but `Text(0) == ""`                   | Chat-tuned model received raw text without its template                                | Use the **Conversation** API instead of `Session.GenerateContent`, or wrap the prompt manually |

