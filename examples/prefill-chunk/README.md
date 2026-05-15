# prefill-chunk

Contrast the engine's default prefill chunking with a caller-selected
`WithPrefillChunkSize`. Both runs are cold — no internal warmup
pass — to match upstream `litert_lm_advanced_main --benchmark`
methodology.

## What this exercises

`WithPrefillChunkSize(n)` breaks the prefill phase into pieces of at
most `n` tokens, processed serially. A 415-token prompt with `n=128`
runs as 3 prefill calls of 128, 128, 159 tokens. `n=-1` disables
chunking (one call). Only honored by the dynamic executor.

CPU-only knob. The option is plumbed but ignored on GPU; the example
prints a notice and proceeds.

## Methodology

Two loads run against the same `-cache-dir`:

1. **Run 1** — default (no `WithPrefillChunkSize` call).
2. **Run 2** — `WithPrefillChunkSize(-chunk)`.

No warmup. Run 1 measures a fully cold load and inference. Run 2
shares the OS process with Run 1; process-resident state persists
across `Close → New` even though the engine handle is fresh. For a
clean per-chunk-size cold reading, invoke twice with different
`-chunk` values and compare the Run 1 numbers.

The built-in prompt tokenizes to 415 tokens on Gemma 4, large enough
that prefill dominates the Generate timing.

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
| `-backend` | `cpu` | `cpu` or `gpu`; `WithPrefillChunkSize` is CPU-only. |
| `-chunk` | `128` | Prefill chunk size for run 2. `-1` disables chunking. |
| `-prompt` | built-in 415-token paragraph | Long prompt that makes prefill dominate. |
| `-cache-dir` | (fresh temp dir) | Directory passed to `WithCacheDir`. Empty creates a temp dir removed at exit. |
| `-keep-cache` | `false` | Skip cleanup of an auto-created temp dir. |
| `-max-tokens` | `4096` | Engine total token budget. |

## Run

```sh
go run ./examples/prefill-chunk -model "$MODEL" -backend cpu
go run ./examples/prefill-chunk -model "$MODEL" -backend cpu -chunk 1024
go run ./examples/prefill-chunk -model "$MODEL" -backend cpu -chunk -1
```

## Observations on Gemma 4 E2B, CPU (Windows 11)

Three invocations on a 415-token prompt:

| Configuration | Run 1 (default) prefill | Run 2 (chunk) prefill | Decode tok/s |
|---|---|---|---|
| `-chunk 128` | 117.7 | 124.2 | 24.3 → 25.3 |
| `-chunk 1024` | 121.9 | 119.0 | 25.2 → 24.8 |
| `-chunk -1` | 117.5 | 116.7 | 24.6 → 24.5 |

All chunk values, including the engine default, land at 117-124
tok/s prefill — a ~6% spread within run-to-run noise. The option is
plumbed but does not change observable throughput on this
configuration. Decode is unaffected (chunking is a prefill-only
knob).

On Gemma 4 E2B the model exposes two prefill signatures
(`prefill_128`, `prefill_1024`). Both runs at 415 tokens appear to
land on the same signature path regardless of the chunk hint.

## Cross-check against `litert_lm_advanced_main --benchmark`

Attempted with `--benchmark_prefill_tokens=415 --benchmark_decode_tokens=75
--max_num_tokens=1024` on the same model and machine, varying
`--prefill_chunk_size`. The native binary's benchmark mode reports
`RunPrefillAsync status: INTERNAL: ERROR` and `Prefill Turns: 0` for
prefill counts above 128 with this model — `--prefill_chunk_size`
cannot be cross-validated against the upstream binary here.

At `--benchmark_prefill_tokens=128 --benchmark_decode_tokens=128
--max_num_tokens=1024` (default chunk), the native binary reports
**597 tok/s prefill** and **33 tok/s decode** on CPU. That number
reflects the engine's synthetic-prompt throughput on a 128-token
input with no chat template; it is not directly comparable to this
example's 415-token templated prompt running through the high-level
`Generate` path.

## Notes

- On GPU the option is a no-op (CPU-only). Differences between Run
  1 and Run 2 on GPU reflect per-process kernel warmup, not the
  chunk knob.
- Caches are model-specific. Switching models within the same
  `-cache-dir` triggers a fresh compile during Run 1.
