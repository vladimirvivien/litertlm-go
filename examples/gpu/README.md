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

1. LiteRT-LM shared library files staged in `LITERTLM_LIB` with
   GPU dependencies satisfied:

   ```
   cp ./LiteRT-LM/bazel-bin/c/liblitertlm_c.so                   $LITERTLM_LIB/
   cp ./LiteRT-LM/prebuilt/libGemmaModelConstraintProvider.so    $LITERTLM_LIB/
   cp ./LiteRT-LM/prebuilt/libLiteRt.so                          $LITERTLM_LIB/
   cp ./LiteRT-LM/prebuilt/libLiteRtWebGpuAccelerator.so         $LITERTLM_LIB/
   cp ./LiteRT-LM/prebuilt/libLiteRtTopKWebGpuSampler.so         $LITERTLM_LIB/
   # Plus any GPU drivers and GPU framework files your platform requires.
   ```

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
