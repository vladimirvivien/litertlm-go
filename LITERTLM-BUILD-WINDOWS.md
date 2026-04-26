# Building the LiteRT-LM C Shared Libraries on Windows

This guide complements [`LITERTLM-BUILD.md`](./LITERTLM-BUILD.md), which only
covers Linux and macOS, and provides steps to build the necessary C API libraries
needed for Go integration. 

The Windows build needs additional Windows-specific adjustments to the local Bazel 
package and one extra runtime step when launching examples.

> Tested on Windows 11 with MSVC 2022, Bazel 7.6.1 (via Bazelisk), Python 3.13,
> Git for Windows + Git LFS, and the JDK. The CPU build was
> verified end-to-end against `gemma-4-E4B-it.litertlm`.

## Prerequisites

Install per [LiteRT-LM's official Windows build
guide](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/getting-started/build-and-run.md#deploy_to_windows):

- Visual Studio 2022 with the MSVC toolchain (installed for all users).
- Git for Windows (includes Git Bash).
- Python 3.13.
- Bazelisk (`winget install --id=Bazel.Bazelisk -e`).
- JDK with `JAVA_HOME` pointing at it.
- `LongPathsEnabled` set to `true` in the registry.

## 1. Clone LiteRT-LM and pull LFS binaries

```powershell
git clone https://github.com/google-ai-edge/LiteRT-LM.git
cd LiteRT-LM
git lfs install --local
git lfs pull
```

Confirm the Windows prebuilts are pulled down:

```powershell
dir prebuilt\windows_x86_64
```

## 2. Create (or update) `c/litertlm_c_api/BUILD`
For windows builds, use the Bazel BUILD shown below 
with configurations added to bypass issues with the 
upstream project: 

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

## 3. Build the CPU shared library

Use the short `--output_base` flag to keep paths shorter to avoid Windows' length limit
issues. Under Git Bash you must also set `MSYS_NO_PATHCONV=1` so the leading `//` 
is not rewritten as a path.

```powershell
$Env:MSYS_NO_PATHCONV = 1

bazelisk --output_base=C:\bzl build //c/litertlm_c_api:litertlm_c_cpu --config=windows
```

Output: `bazel-bin\c\litertlm_c_api\litertlm_c_cpu.dll` (~12 MB). 

Verify the C API is exported:

```bash
objdump -p bazel-bin/c/litertlm_c_api/litertlm_c_cpu.dll | grep -E "litert_lm_engine_create|litert_lm_set_min_log_level"
```

You should see ~400 exports including `litert_lm_engine_create`,
`litert_lm_session_config_create`, `litert_lm_set_min_log_level`, etc.


## 4. Stage the libraries (CPU)

Copy the shared library files in a known location `$Env:LITERTLM_LIB`
- Copy the pulled prebuilt library files from `prebuilt/<os_arc_dir>`
- Copy the freshly built CPU API library `litertlm_c_cpu.dll`

```powershell
$Env:LITERTLM_LIB = "$Env:USERPROFILE\include\litertlm\lib"
mkdir $Env:LITERTLM_LIB -Force | Out-Null

# Prebuilt runtime DLLs (lib-prefixed names, exactly as shipped).
copy prebuilt\windows_x86_64\*.dll $Env:LITERTLM_LIB\

# Freshly-built C API DLL 
copy bazel-bin\c\litertlm_c_api\litertlm_c_cpu.dll $Env:LITERTLM_LIB\
```

## 5. Run an example


```powershell
$Env:LITERTLM_LIB = "$Env:USERPROFILE\include\litertlm\lib"
go run .\examples\hello -model C:\path\to\gemma-4-E4B-it.litertlm
```

## 6. Build GPU shared libraries

Building the GPU C shared libraries require some additional flags.
The build will produce `bazel-bin\c\litertlm_c_api\litertlm_c.dll`
(instead of `litertlm_c_cpu.dll`). Again, use the short `--output_base` 
flag to keep paths shorter to avoid Windows' length limit issues. 
The Bazel command adds `--define` flag to futher configure the build.


### Build

```powershell
$Env:MSYS_NO_PATHCONV = 1

bazelisk --output_base=C:\bzl build //c/litertlm_c_api:litertlm_c `
    --config=windows `
    --define=litert_link_capi_so=true `
    --define=resolve_symbols_in_exec=false
```

### Stage GPU runtime files

For GPU support on Windows, the DirectX Shader Compiler is required at runtime; 
it ships with the Windows 10/11 SDK at:

```
C:\Program Files (x86)\Windows Kits\10\bin\<sdk-version>\x64\
```

Next, copy the required shared libraries to directory `LITERTLM_LIB`:

```powershell
$Env:LITERTLM_LIB = "$Env:USERPROFILE\include\litertlm\lib-gpu"
mkdir $Env:LITERTLM_LIB -Force | Out-Null

# All prebuilt runtime DLLs (including libLiteRt.dll — the prebuilt one!).
copy prebuilt\windows_x86_64\*.dll $Env:LITERTLM_LIB\

# Freshly-built GPU C API DLL.
copy bazel-bin\c\litertlm_c_api\litertlm_c.dll $Env:LITERTLM_LIB\

# DirectX Shader Compiler (adjust SDK version to match what's installed).
$SDK = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"
copy "$SDK\dxcompiler.dll" $Env:LITERTLM_LIB\
copy "$SDK\dxil.dll"       $Env:LITERTLM_LIB\
```

### Run with GPU

```powershell
$Env:LITERTLM_LIB = $Env:LITERTLM_LIB
go run .\examples\chat -model C:\path\to\gemma-4-E4B-it.litertlm -backend gpu
```

On boot you should see WebGPU adapter selection logs naming your discrete
GPU (e.g. `Selected adapter: NVIDIA GeForce RTX 4070 Laptop GPU, ...
backend=Direct3D 12`) followed by normal chat output.

## Verified examples

| Example | Backend | Status |
|---|---|---|
| `hello` | `cpu` | ✅ Synchronous Session API works |
| `chat`  | `cpu` | ✅ Conversation API two-turn demo works |
| `chat`  | `gpu` | ✅ NVIDIA RTX 4070 via WebGPU/D3D12 |
| `stream` | `cpu` | ✅ Token-by-token streaming works |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `load: ... lib<name>.dll: error loading library: The specified module could not be found.` | Plugin DLL missing from `$LITERTLM_LIB`, or you copied the unprefixed name | Re-copy `prebuilt\windows_x86_64\*.dll` (lib-prefixed) into `$LITERTLM_LIB` — the wrapper preloads them under their original `lib<name>.dll` filenames |
| `load: could not load "litert_lm_*": The specified procedure could not be found.` | Built without `/WHOLEARCHIVE` and/or `/EXPORT` linkopts | Apply step 2 and rebuild |
| `objdump -p ... litertlm_c_cpu.dll` shows ~357 `LiteRtDispatch*` exports but no `litert_lm_*` | Same as above | Same as above |
| `ERROR: Skipping '/c:litertlm_c_cpu': invalid package name '/c'` (Git Bash) | MSYS rewrote `//c:...` as a Windows path | `MSYS_NO_PATHCONV=1 bazelisk build //c/litertlm_c_api:litertlm_c_cpu --config=windows` |
| Build fails with `LongPathsEnabled` errors | NTFS long path support disabled | Enable it in the registry, or use a shorter `--output_base` (e.g. `C:\bzl`) |
| `cp: cannot create regular file ...: Permission denied` when restaging | Bazel marks outputs read-only | `chmod u+w` the destination first, or use `copy /Y` in PowerShell |
| GPU run crashes with `Exception 0xc0000005` after `delegate_webgpu.cc:644 # of threads to compile kernels = 1` | Staging used the freshly-built `libLiteRt.dll` from `bazel-bin\` instead of the prebuilt one (ABI mismatch with the prebuilt accelerator plugins) | Replace with `prebuilt\windows_x86_64\libLiteRt.dll` |
| GPU run logs `WARNING: GPU accelerator could not be loaded and registered` then falls back to CPU | One of `libLiteRtWebGpuAccelerator.dll` / `libLiteRtTopKWebGpuSampler.dll` / DXC missing from `$LITERTLM_LIB_GPU` | Re-run the §6 staging step |
