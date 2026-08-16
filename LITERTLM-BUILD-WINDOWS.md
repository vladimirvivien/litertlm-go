# Staging & Building LiteRT-LM C Shared Libraries on Windows

`litertlm-go` loads native LiteRT-LM shared libraries (`*.dll`) at runtime. Starting with **v0.16.0**, official prebuilt Windows binaries are distributed by the LiteRT-LM project.

---

## 1. Staging Prebuilt Libraries on Windows (Recommended)

### Option A: Automated Provisioning via `litertlm.LibFetch`

`litertlm-go` includes a built-in helper that downloads the official Windows prebuilt archive, extracts the required DLLs, and locates DirectX Shader Compiler (DXC) dependencies automatically:

```go
package main

import (
    "fmt"
    "log"

    "github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
    libDir, err := litertlm.LibFetch("windows", "amd64", "v0.16.0")
    if err != nil {
        log.Fatalf("LibFetch failed: %v", err)
    }
    fmt.Printf("LiteRT-LM libraries staged at: %s\n", libDir)
}
```

### Option B: Manual Download & Staging

1. Download `litertlm-windows-x86_64-v0.16.0.zip` from the [LiteRT-LM Releases](https://github.com/google-ai-edge/LiteRT-LM/releases/tag/v0.16.0).
2. Extract the archive into your staging directory (e.g. `%USERPROFILE%\include\litertlm\lib`):
   ```powershell
   $Env:LITERTLM_LIB = "$Env:USERPROFILE\include\litertlm\lib"
   New-Item -ItemType Directory -Path $Env:LITERTLM_LIB -Force | Out-Null
   Expand-Archive -Path litertlm-windows-x86_64-v0.16.0.zip -DestinationPath $Env:LITERTLM_LIB -Force
   ```
3. **DirectX Shader Compiler (DXC) for GPU Inference:**
   For Direct3D 12 / WebGPU execution, copy `dxcompiler.dll` and `dxil.dll` from the Windows SDK:
   ```powershell
   $SDK = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64"
   if (Test-Path "$SDK\dxcompiler.dll") {
       Copy-Item "$SDK\dxcompiler.dll" $Env:LITERTLM_LIB\
       Copy-Item "$SDK\dxil.dll"       $Env:LITERTLM_LIB\
   }
   ```

---

## 2. Testing Your Setup

Download a `.litertlm` model file (such as `gemma3-1b-it-int4.litertlm`) from Hugging Face's [LiteRT Community](https://huggingface.co/litert-community), then run an example:

```powershell
# Set library path
$Env:LITERTLM_LIB = "$Env:USERPROFILE\include\litertlm\lib"

# CPU Inference
go run .\examples\hello -model C:\path\to\gemma3-1b-it-int4.litertlm -backend cpu

# GPU Inference (WebGPU / Direct3D 12)
$Env:PATH = "$Env:LITERTLM_LIB;$Env:PATH"
$Env:LLVM_PROFILE_FILE = "NUL"
go run .\examples\chat -model C:\path\to\gemma3-1b-it-int4.litertlm -backend gpu
```

---

## 3. Building from Source with MSVC & Bazel (Advanced / Custom Builds)

If you are developing custom C++ engine features or operators, compile from source on Windows:

### Prerequisites

* Visual Studio 2022 (MSVC C++ x64/x86 build tools).
* Git for Windows (with Git Bash and Git LFS).
* Python 3.11+.
* Bazelisk on system `PATH`.
* Enable `LongPathsEnabled` in the Windows Registry (`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`).

### Step 1: Clone Repository

```powershell
git clone https://github.com/google-ai-edge/LiteRT-LM.git
cd LiteRT-LM
git checkout v0.16.0
git lfs install --local
git lfs pull
```

### Step 2: Configure Bazel Target

Create `c/litertlm_c_api/BUILD`:

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

### Step 3: Build Binaries

```powershell
$Env:MSYS_NO_PATHCONV = 1

# CPU Library
bazelisk --output_base=C:\bzl build //c/litertlm_c_api:litertlm_c_cpu --config=windows

# GPU Library
bazelisk --output_base=C:\bzl build //c/litertlm_c_api:litertlm_c `
    --config=windows `
    --define=litert_runtime_link_mode=dynamic `
    --define=resolve_symbols_in_exec=false
```

### Step 4: Stage Binaries

```powershell
$Env:LITERTLM_LIB = "$Env:USERPROFILE\include\litertlm\lib"
New-Item -ItemType Directory -Path $Env:LITERTLM_LIB -Force | Out-Null

Copy-Item prebuilt\windows_x86_64\*.dll $Env:LITERTLM_LIB\
Copy-Item bazel-bin\c\litertlm_c_api\litertlm_c*.dll $Env:LITERTLM_LIB\
```

---

## 4. Troubleshooting

| Symptom | Cause | Solution |
|---|---|---|
| `load: ... lib<name>.dll: The specified module could not be found` | Auxiliary plugin DLLs missing from staging path | Ensure all `prebuilt\windows_x86_64\*.dll` files are copied to `$Env:LITERTLM_LIB` |
| `DynamicLib.Open: dxil.dll Windows Error: 87` / `Failed to create WebGPU environment` | `dxcompiler.dll` or `dxil.dll` missing | Copy both DLLs from the Windows SDK into `$Env:LITERTLM_LIB` |
| Build fails with `LongPathsEnabled` errors | NTFS long paths disabled | Enable LongPathsEnabled in registry or use `--output_base=C:\bzl` |
| Empty `default.profraw` files appear in current directory | Upstream binary LLVM profiling runtime | Set `$Env:LLVM_PROFILE_FILE = "NUL"` in PowerShell |
