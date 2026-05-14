# activation-dtype

Contrast the engine's default activation precision with a
caller-selected `WithActivationDataType`. Both runs are cold —
no internal warmup pass — to match upstream
`litert_lm_advanced_main --benchmark` methodology.

## What this exercises

`WithActivationDataType(t)` selects the activation precision for the
engine. Per `c/engine.h`:

| Value | Type |
|---|---|
| 0 | F32 |
| 1 | F16 |
| 2 | I16 |
| 3 | I8 |

Effects are backend-dependent.

## Methodology

Two loads run against the same `-cache-dir`:

1. **Run 1** — default activation dtype (no option set).
2. **Run 2** — `WithActivationDataType(-dtype)`.

No warmup pass is run. Run 1 measures a fully cold load and
inference. Run 2 shares the OS process with Run 1, so some library
and GPU initialization state persists across the `Close → New`
boundary even though the engine handle is fresh. When Run 2's dtype
matches Run 1's, the engine's kernel state is also hot and Run 2's
prefill numbers are much faster than its dtype would justify in
isolation. For a clean per-dtype cold reading, invoke the binary
twice with different `-dtype` values and compare the Run 1 numbers.

`WithBenchmarkEnabled` is set on both loads. `WithCacheDir` is set
so on-disk XNNPACK / mldrift caches survive between Run 1 and Run 2.

## Prerequisites

- `.litertlm` model file. Gemma 4 family works.
- LiteRT-LM shared libraries on disk. Pass `-lib <dir>` or set
  `LITERTLM_LIB=<dir>`.

## Flags

| Flag | Default | Description |
|---|---|---|
| `-model` | (required) | Path to `.litertlm` model file. |
| `-lib` | `$LITERTLM_LIB` | Directory holding the shared libraries. |
| `-backend` | `cpu` | `cpu` or `gpu`. |
| `-dtype` | `1` | Activation dtype for run 2 (0..3). |
| `-prompt` | Go-popularity prompt | Completion-style prompt issued in both runs. |
| `-cache-dir` | (fresh temp dir) | Directory passed to `WithCacheDir`. Empty creates a temp dir removed at exit. |
| `-keep-cache` | `false` | Skip cleanup of an auto-created temp dir. |

## Run

```sh
go run ./examples/activation-dtype -model "$MODEL" -backend cpu
go run ./examples/activation-dtype -model "$MODEL" -backend gpu
go run ./examples/activation-dtype -model "$MODEL" -backend cpu -dtype 0
```

## Observations on Gemma 4 E2B

### CPU (Windows 11)

All four dtype values and the default produce the same numbers
within run-to-run noise. The CPU XNNPACK kernels for Gemma 4 appear
to use a fixed accumulation precision regardless of the dtype hint.

| Configuration (cold) | Prefill tok/s | Decode tok/s | TTFT |
|---|---|---|---|
| Run 1 default | 31.4 – 35.4 | 22.7 – 24.5 | 436 – 490 ms |
| Run 2 `-dtype 0` (F32) | 33.6 | 24.8 | 457 ms |
| Run 2 `-dtype 1` (F16) | 35.3 | 25.3 | 436 ms |

### GPU (DirectX / WebGPU, Windows 11)

`-dtype 1` (F16) invocation:

| | Prefill tok/s | Decode tok/s | TTFT | Decoded tokens |
|---|---|---|---|---|
| Run 1 default (cold) | 16.5 | 73.4 | 861 ms | 4082 |
| Run 2 F16 (process-warm; dtype matches Run 1) | 394.0 | 72.5 | 49 ms | 4082 |

`-dtype 0` (F32) invocation:

| | Prefill tok/s | Decode tok/s | TTFT | Decoded tokens |
|---|---|---|---|---|
| Run 1 default = F16 (cold) | 13.6 | 70.1 | 1040 ms | 4082 |
| Run 2 F32 (cold dtype kernels in process-warm state) | 10.7 | 52.3 | 1329 ms | 38 |

Run 1 cold-default at 14 tokens lands at 14-17 tok/s prefill across
both invocations. Run 2 F16 reaches 394 tok/s because its kernels
were compiled by Run 1; Run 2 F32 stays at 10.7 tok/s because its
F32 kernels are cold even though the process is warm. F32 decoded
38 tokens to a natural stop where F16/default looped to the 4096
cap — a behavioral difference visible in the decoded-token column.

## Cross-check against upstream `litert_lm_advanced_main --benchmark`

Same model and machine, `--benchmark_prefill_tokens=128
--benchmark_decode_tokens=128 --max_num_tokens=1024`, fresh
`--cache_dir` per invocation:

| Backend | Configuration | Prefill tok/s | Decode tok/s |
|---|---|---|---|
| CPU | default (F16) | 597 | 33 |
| CPU | `--force_f32` | 577 | 34 |
| GPU | default (F16) | 146 | 76 |
| GPU | `--force_f32` | 70 | 78 |

The upstream binary uses synthetic 128-token prefills with no chat
template and no per-call overhead, so per-token throughput at 128
tokens exceeds the per-token rate seen here on a 14-token templated
prompt. On GPU the F16/F32 prefill ratio at 128 cold tokens is
~2.1×.

For honest dtype throughput characterization on a given backend,
prefer the upstream binary's `--benchmark` mode with a token count
that matches the production workload.

## Notes

- If the requested dtype is unsupported on the selected backend,
  `litertlm.New` returns an error; the binary prints
  `engine construction failed: ...` and exits 0.
- Caches are model-specific. Switching models within the same
  `-cache-dir` triggers a fresh compile during Run 1.
