# gpu — GPU-accelerated inference + benchmark readout

Runs the same streaming inference as [`stream`](../stream/), but with
`backend="gpu"` and `EnableBenchmark()` so the example also prints
per-turn prefill / decode throughput and time-to-first-token.

## What this example shows

- Setting `backend = "gpu"` when calling `NewEngineSettings`.
- Turning on benchmark collection with `settings.EnableBenchmark()`.
- Reading metrics off `BenchmarkInfo`: `TotalInitTime`, `TimeToFirstToken`,
  `PrefillTokensPerSec(turn)`, `DecodeTokensPerSec(turn)`.
- The runtime distribution shape required for GPU (which differs from the
  CPU build).

## Prerequisites

> The CPU examples Just Work after a single `bazel build //c:litertlm_c_cpu`.
> GPU has more moving parts — read this section carefully.

### 1. Build the GPU-capable C-API library

```bash
bazel build //c:litertlm_c \
    --define=litert_link_capi_so=true \
    --define=resolve_symbols_in_exec=false
```

This produces `bazel-bin/c/liblitertlm_c.{so,dylib}` (or
`bazel-bin\c\litertlm_c.dll` on Windows). See `build_litertlm.md` §3 for
full per-OS instructions.

### 2. Stage the GPU runtime files into one directory

In addition to the wrapper library, the GPU build dynamically loads four
plugins from the `prebuilt/<os>/` directory of the LiteRT-LM checkout.
Per `build_litertlm.md` §4.4, on Linux x86_64:

```bash
PREBUILT=$LITERTLM/prebuilt/linux_x86_64
DIST=./dist-gpu/lib
mkdir -p $DIST
cp $LITERTLM/bazel-bin/c/liblitertlm_c.so          $DIST/
cp $PREBUILT/libGemmaModelConstraintProvider.so    $DIST/
cp $PREBUILT/libLiteRt.so                          $DIST/
cp $PREBUILT/libLiteRtWebGpuAccelerator.so         $DIST/
cp $PREBUILT/libLiteRtTopKWebGpuSampler.so         $DIST/
```

macOS arm64 also requires `libLiteRtMetalAccelerator.dylib`. Windows
also requires the **DirectX Shader Compiler** (`dxcompiler.dll` +
`dxil.dll`) on the system. See `build_litertlm.md` §4.1 for the full
table of which platform needs what.

### 3. Bridge the wrapper's expected name to the GPU build

The litertlm-go wrapper opens the main library by the short name
`litertlm_c_cpu` (resolved per platform to `liblitertlm_c_cpu.so` /
`.dylib` / `litertlm_c_cpu.dll`). For the wrapper to load the **GPU**
build, you have two options:

**Option A — symlink (Linux/macOS):**

```bash
cd dist-gpu/lib
ln -sf liblitertlm_c.so liblitertlm_c_cpu.so          # Linux
ln -sf liblitertlm_c.dylib liblitertlm_c_cpu.dylib    # macOS
```

```powershell
# Windows (administrator shell)
cd dist-gpu\lib
mklink litertlm_c_cpu.dll litertlm_c.dll
```

**Option B — rename the file:**

```bash
mv dist-gpu/lib/liblitertlm_c.so dist-gpu/lib/liblitertlm_c_cpu.so
```

A future revision of the Go wrapper may add a `LITERTLM_LIB_NAME`
override; for now the bridge step above is the supported path.

### 4. A `.litertlm` model file and Go 1.22+

Same as the other examples.

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist-gpu/lib \
    go run ./examples/gpu \
    -model /abs/path/to/gemma-4-E2B-it.litertlm
```

| Flag        | Default                                                            |
| ----------- | ------------------------------------------------------------------ |
| `-model`    | (required)                                                         |
| `-prompt`   | `"Summarise Go's approach to concurrency in one paragraph."`       |
| `-max`      | `1024`                                                             |
| `-lib`      | `$LITERTLM_LIB`                                                    |

There is **no `-backend` flag** — this example is GPU-only by design.

## Expected output

The streamed text first, then a benchmark block:

```
Go's approach to concurrency centers on goroutines and channels: lightweight
threads scheduled by the runtime that communicate over typed pipes …
--- GPU benchmark ---
Total init time:       3.214 s
Time to first token:   0.241 s
Prefill turn 0:        980.4 tokens/sec (16 tokens)
Decode  turn 0:        85.7 tokens/sec (132 tokens)
```

Compare against the CPU [`stream`](../stream/) example on the same
machine to see the GPU speedup on prefill (often 5–30×) and decode
(often 2–5×).

## Troubleshooting

| Symptom                                                                | Cause                                                          | Fix                                                                                            |
| ---------------------------------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `load: ...liblitertlm_c_cpu.so: cannot open shared object file`        | Step 3 above wasn't done                                       | Symlink or rename the GPU lib so the short name matches.                                       |
| `engine_create failed` with GPU plugin warnings                        | One or more of the four GPU prebuilts is missing from `LITERTLM_LIB` | Re-check step 2; `ls $LITERTLM_LIB/libLiteRt*.so` should show 3 files (Linux).                 |
| Inference falls back to CPU silently                                   | GPU plugins loaded but the device couldn't initialise          | Check stderr — there will be `WARNING: GPU accelerator could not be loaded and registered.`    |
| Windows: GPU build fails to load                                       | DirectX Shader Compiler missing                                | Install `dxcompiler.dll` + `dxil.dll` (Windows SDK or DirectXShaderCompiler GitHub release).   |
