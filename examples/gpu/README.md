# Example: Use GPU-Accelerated Inference with Benchmark

This example uses the low-level `litertlm-go` API to run GPU-accelerated inference wiht `backend="gpu"` 
and uses `EnableBenchmark()` to capture and print per-turn prefill / decode throughput and time-to-first-token.

## What this example shows

- Use `litertlm.Load()` to load the LiteRT-LM libraries with `"gpu"` backend.
- Manually configure: 
  - Logging (`litertlm.SetMinLogLevel()`) 
  - Instantiating new engine settings (`litertlm.NewEngineSettings()`)
  - Turn on benchmark collection with `settings.EnableBenchmark()`
  - Initialize a new instance of the LiteRT-LM runtime (`litertlm.NewEngine()`)
- At runtime, capture metrics `BenchmarkInfo` (`TotalInitTime`, `TimeToFirstToken`,
  `PrefillTokensPerSec(turn)`, `DecodeTokensPerSec(turn)`)

## Prerequisites

1. LiteRT-LM shared library files staged in`LITERTLM_LIB` with GPU-dependencies satisfied.
  ```
  cp ./LiteRT-LM/bazel-bin/c/liblitertlm_c.so                   $LITERTLM_LIB/
  cp ./LiteRT-LM/prebuilt/libGemmaModelConstraintProvider.so    $LITERTLM_LIB/
  cp ./LiteRT-LM/prebuilt/libLiteRt.so                          $LITERTLM_LIB/
  cp ./LiteRT-LM/prebuilt/libLiteRtWebGpuAccelerator.so         $LITERTLM_LIB/
  cp ./LiteRT-LM/prebuilt/libLiteRtTopKWebGpuSampler.so         $LITERTLM_LIB/
  # Also, you must stage GPU drivers and GPU framework files in $LITERTLM_LIB/
  ```
2. A `.litertlm` model (i.e. Gemma 4). 
3. `litertlm-go`

## Run

```bash
LITERTLM_LIB=/abs/path/to/lib \
    go run ./examples/gpu \
    -model /abs/path/to/gemma-4-E2B-it.litertlm
```

| Flag        | Default                                                            |
| ----------- | ------------------------------------------------------------------ |
| `-model`    | (required)                                                         |
| `-prompt`   | `"Summarise Go's approach to concurrency in one paragraph."`       |
| `-max`      | `1024`                                                             |
| `-lib`      | `$LITERTLM_LIB`                                                    |

## Expected output

```
Go's approach to concurrency centers on goroutines and channels: lightweight
threads scheduled by the runtime that communicate over typed pipes …
--- GPU benchmark ---
Total init time:       3.214 s
Time to first token:   0.241 s
Prefill turn 0:        980.4 tokens/sec (16 tokens)
Decode  turn 0:        85.7 tokens/sec (132 tokens)
```