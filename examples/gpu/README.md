# Example: GPU-Accelerated Inference

Loads the GPU-capable LiteRT-LM build, configures an engine for the
GPU backend, and streams tokens from a single prompt.

For benchmark collection (init time, time-to-first-token, per-turn
prefill / decode throughput), see
[`examples/benchmarks/`](../benchmarks).

## What this example shows

- Loading the GPU LiteRT-LM library with `litertlm.Load(lib, "gpu", "")`.
- Manual configuration via the low-level API:
  - Logging (`litertlm.SetMinLogLevel`)
  - Engine settings (`litertlm.NewEngineSettings` with `backend="gpu"`)
  - Engine and session lifecycle (`NewEngine`, `engine.NewSession`)
- Streaming tokens with `Session.GenerateContentStreamCh`.

## Prerequisites

1. LiteRT-LM shared library staged in `LITERTLM_LIB` (or fetched with `-get-lib v0.19.1`):

   - Linux: `liblitert-lm.so`
   - macOS: `liblitert-lm.dylib`
   - Windows: `litert-lm.dll` plus DirectX Shader Compiler (`dxcompiler.dll`, `dxil.dll`)

2. A `.litertlm` chat-tuned model (e.g. Gemma 4).

## Run

```bash
LITERTLM_LIB=/abs/path/to/lib \
    go run ./examples/gpu \
    -model /abs/path/to/gemma-4-E2B-it.litertlm
```

| Flag      | Default                                                       |
| --------- | ------------------------------------------------------------- |
| `-model`  | (required)                                                    |
| `-prompt` | `"Summarise Go's approach to concurrency in one paragraph."`  |
| `-max`    | `1024`                                                        |
| `-lib`    | `$LITERTLM_LIB`                                               |

## Expected output

```
Go's approach to concurrency centers on goroutines and channels:
lightweight threads scheduled by the runtime that communicate over
typed pipes …
```
