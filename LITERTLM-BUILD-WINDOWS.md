# Staging & Building LiteRT-LM C Shared Libraries on Windows

`litertlm-go` loads the unified `litert-lm.dll` shared library targeting LiteRT-LM **v0.19.1** (C API `1.0.0`). Official prebuilt Windows binaries are distributed by the LiteRT-LM project.

---

## 1. Staging Prebuilt Libraries on Windows (Recommended)

### Option A: Automated Provisioning via `litertlm.FetchLib`

`litertlm-go` includes a built-in helper that downloads the official Windows prebuilt archive (`litert_lm_c_api-1.0.0.zip`) and extracts `litert-lm.dll`. Use `libfetch.Fetch` with `libfetch.WithBackend("gpu")` when you also want DirectX Shader Compiler (`dxcompiler.dll`, `dxil.dll`) staged automatically:

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
    libDir, err := litertlm.FetchLib(context.Background(), "windows", "amd64", "v0.19.1")
    if err != nil {
        log.Fatalf("FetchLib failed: %v", err)
    }
    fmt.Printf("LiteRT-LM libraries staged at: %s\n", libDir)
}
```

### Option B: Manual Download & Staging

1. Download `litert_lm_c_api-1.0.0.zip` from [LiteRT-LM Releases](https://github.com/google-ai-edge/LiteRT-LM/releases/tag/v0.19.1).
2. Extract the archive and copy `lib\windows_x86_64\litert-lm.dll` into your staging directory (e.g. `$Env:USERPROFILE\include\litertlm\lib`):
   ```powershell
   $Env:LITERTLM_LIB = "$Env:USERPROFILE\include\litertlm\lib"
   New-Item -ItemType Directory -Path $Env:LITERTLM_LIB -Force | Out-Null
   Expand-Archive -Path litert_lm_c_api-1.0.0.zip -DestinationPath .\litert_lm_c_api -Force
   Copy-Item .\litert_lm_c_api\lib\windows_x86_64\litert-lm.dll $Env:LITERTLM_LIB\
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
git checkout v0.19.1
git lfs install --local
git lfs pull
```

### Step 2: Build the Unified DLL

Upstream defines the unified C-API shared library target `//c:litert-lm`:

```powershell
$Env:MSYS_NO_PATHCONV = 1
bazelisk --output_base=C:\bzl build //c:litert-lm --config=windows
```

### Step 3: Stage Binaries

```powershell
$Env:LITERTLM_LIB = "$Env:USERPROFILE\include\litertlm\lib"
New-Item -ItemType Directory -Path $Env:LITERTLM_LIB -Force | Out-Null

Copy-Item bazel-bin\c\litert-lm.dll $Env:LITERTLM_LIB\
```

---

## 4. Troubleshooting

| Symptom | Cause | Solution |
|---|---|---|
| `"litert-lm" not found` or `load: ... The specified module could not be found` | `litert-lm.dll` missing from staging path | Ensure `litert-lm.dll` is in `$Env:LITERTLM_LIB` (or pass `WithLib`) |
| `DynamicLib.Open: dxil.dll Windows Error: 87` / `Failed to create WebGPU environment` | `dxcompiler.dll` or `dxil.dll` missing | Copy both DLLs from the Windows SDK into `$Env:LITERTLM_LIB` |
| Build fails with `LongPathsEnabled` errors | NTFS long paths disabled | Enable LongPathsEnabled in registry or use `--output_base=C:\bzl` |
| Empty `default.profraw` files appear in current directory | Upstream binary LLVM profiling runtime | Set `$Env:LLVM_PROFILE_FILE = "NUL"` in PowerShell |
