# Building the LiteRT-LM C Shared Libraries on Windows

This guide complements [`LITERTLM-BUILD.md`](./LITERTLM-BUILD.md) (which only
covers Linux and macOS) and provides steps to build the necessary C API libraries
to use on the Windows operating system. 

## Prerequisites

Install your pre-requisites per [LiteRT-LM's official Windows build
guide](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/getting-started/build-and-run.md#deploy_to_windows):

- Visual Studio 2022 with the MSVC toolchain (installed for all users).
- Git for Windows (includes Git Bash).
- Python 3.13.
- Bazelisk
- JDK 
- Don't forget `LongPathsEnabled` set to `true` in the registry.

## 1. Clone the LiteRT-LM repo

```powershell
git clone https://github.com/google-ai-edge/LiteRT-LM.git
cd LiteRT-LM
git lfs install --local
git lfs pull
```

Confirm the Windows prebuilt dependencies are pulled down:

```powershell
dir prebuilt\windows_x86_64
```

## 2. Create (or update) Bazel BUILD file
Create Bazel file `c/litertlm_c_api/BUILD` if you haven't done so.
For windows builds, use the Bazel BUILD snippet shown below 
(with some additional configurations added to bypass issues with the 
upstream project): 

```python
package(default_visibility = ["//visibility:public"])

WIN_EXPORTS = [
    "/EXPORT:litert_lm_log",
    "/EXPORT:litert_lm_set_min_log_level",
]

cc_binary(
    name = "litertlm_c_cpu",
    linkshared = 1,
    linkopts = select({
        "@platforms//os:windows": ["/WHOLEARCHIVE:engine_cpu.lib"] + WIN_EXPORTS,
        "//conditions:default": [],
    }),
    deps = ["//c:engine_cpu"],
)

cc_binary(
    name = "litertlm_c",
    linkshared = 1,
    linkopts = select({
        "@platforms//os:windows": ["/WHOLEARCHIVE:engine.lib"] + WIN_EXPORTS,
        "//conditions:default": [],
    }),
    deps = ["//c:engine"],
)
```

## 3. Build the shared library

Use s short `--output_base` flag to keep paths shorter to avoid Windows' length limit
issues.You must also set `MSYS_NO_PATHCONV=1` so the leading `//` is not rewritten as a path.

```powershell
$Env:MSYS_NO_PATHCONV = 1

# build CPU-only
bazelisk --output_base=C:\bzl build //c/litertlm_c_api:litertlm_c_cpu --config=windows

# build GPU
bazelisk --output_base=C:\bzl build //c/litertlm_c_api:litertlm_c `
    --config=windows 
    --define=litert_link_capi_so=true 
    --define=resolve_symbols_in_exec=false
```

By default, the built files are stored at `bazel-bin/c/litertlm_c_api/*.dll` on Windows.

If you need to start over with a clean build, use `bazelisk clean --expunge` to clear previous builds.

## 4. Stage the library files

Next, store all library files in a known location (`$env:LITERTLM_LIB`) to make them easy to find:

```powershell
# Create staging directory
$Env:LITERTLM_LIB = "$Env:USERPROFILE\include\litertlm\lib"
mkdir $Env:LITERTLM_LIB -Force | Out-Null

# Copy the prebuilt runtime DLLs 
copy prebuilt\windows_x86_64\*.dll $Env:LITERTLM_LIB\

# Copy the newly-built C API DLL (CPU-only)
copy bazel-bin\c\litertlm_c_api\litertlm_c_cpu.dll $Env:LITERTLM_LIB\
```

For GPU use copy the followings:

```powershell
# Copy the C API DLL for GPU use 
copy bazel-bin\c\litertlm_c_api\litertlm_c.dll $Env:LITERTLM_LIB\

# Additionally, copy DirectX Shader Compiler (adjust SDK version for your environment).
$SDK = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"
copy "$SDK\dxcompiler.dll" $Env:LITERTLM_LIB\
copy "$SDK\dxil.dll"       $Env:LITERTLM_LIB\
```

## 5. Run an example
Run the examples to validate your setup.

CPU inference:

```powershell
$Env:LITERTLM_LIB = "$Env:USERPROFILE\include\litertlm\lib"
go run .\examples\hello -model C:\path\to\gemma-4-E4B-it.litertlm -backend cpu
```

GPU-backed inference:

```powershell
$Env:LITERTLM_LIB = "$Env:USERPROFILE\include\litertlm\lib-gpu"
$Env:LLVM_PROFILE_FILE = "NUL"
go run .\examples\chat -model C:\path\to\gemma-4-E4B-it.litertlm -backend gpu
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `load: ... lib<name>.dll: error loading library: The specified module could not be found.` | Plugin DLL missing from `$LITERTLM_LIB`, or you copied the unprefixed name | Re-copy `prebuilt\windows_x86_64\*.dll` (lib-prefixed) into `$LITERTLM_LIB` — the wrapper preloads them under their original `lib<name>.dll` filenames |
| `litertlm: New: ...\litertlm_c_cpu.dll: error loading library: The specified module could not be found.` | Only the GPU/full build (`litertlm_c.dll`) was staged; `-backend cpu` looks for `litertlm_c_cpu.dll` | Either build `litertlm_c_cpu` (§3) and copy it in, or hardlink the GPU build under the CPU name (see §6 *One staging dir for both backends*) |
| `load: could not load "litert_lm_*": The specified procedure could not be found.` | Built without `/WHOLEARCHIVE` and/or `/EXPORT` linkopts | Apply step 2 and rebuild |
| `objdump -p ... litertlm_c_cpu.dll` shows ~357 `LiteRtDispatch*` exports but no `litert_lm_*` | Same as above | Same as above |
| `ERROR: Skipping '/c:litertlm_c_cpu': invalid package name '/c'` (Git Bash) | MSYS rewrote `//c:...` as a Windows path | `MSYS_NO_PATHCONV=1 bazelisk build //c/litertlm_c_api:litertlm_c_cpu --config=windows` |
| Build fails with `LongPathsEnabled` errors | NTFS long path support disabled | Enable it in the registry, or use a shorter `--output_base` (e.g. `C:\bzl`) |
| `cp: cannot create regular file ...: Permission denied` when restaging | Bazel marks outputs read-only | `chmod u+w` the destination first, or use `copy /Y` in PowerShell |
| GPU run crashes with `Exception 0xc0000005` after `delegate_webgpu.cc:644 # of threads to compile kernels = 1` | Staging used the freshly-built `libLiteRt.dll` from `bazel-bin\` instead of the prebuilt one (ABI mismatch with the prebuilt accelerator plugins) | Replace with `prebuilt\windows_x86_64\libLiteRt.dll` |
| GPU run logs `WARNING: GPU accelerator could not be loaded and registered` then falls back to CPU | One of `libLiteRtWebGpuAccelerator.dll` / `libLiteRtTopKWebGpuSampler.dll` / DXC missing from `$LITERTLM_LIB` | Re-run the §6 staging step |
| GPU run aborts in `engine_create` with `DynamicLib.Open: dxil.dll Windows Error: 87` / `Failed to create WebGPU environment` | `dxcompiler.dll` and/or `dxil.dll` not staged into `$LITERTLM_LIB` (these ship with the Windows SDK, not the LiteRT-LM prebuilts) | Copy both from `C:\Program Files (x86)\Windows Kits\10\bin\<sdk-version>\x64\` per §6 *Stage GPU runtime files* |
| Empty `default.profraw` files appear in your working directory after each run | The prebuilt LiteRT-LM deps inherit LLVM `-fprofile-instr-generate` instrumentation; the embedded `__llvm_profile_*` runtime writes a coverage dump to `.\default.profraw` on exit | Set `LLVM_PROFILE_FILE=NUL` in the environment to discard the dump: `$Env:LLVM_PROFILE_FILE = "NUL"; go run …` |
