# parallel-load

Measure `litertlm.New` wall-clock under the engine's default parallel
section loading vs the serial path forced by
`WithParallelSectionLoading(false)`. One mode per invocation; invoke
twice with different `-parallel` values and compare.

## What this exercises

`WithParallelSectionLoading(bool)` toggles parallel deserialization
of sections within the `.litertlm` container (weights, tokenizer,
decode and prefill graphs, multimodal adapters) during `New`. The C
side defaults to `true`. Passing `false` forces serial reads.

The effect is observable only in `litertlm.New` wall-clock; runtime
throughput after construction is unaffected.

## Methodology

Single load per invocation. Each invocation uses a fresh cache dir
(`-cache-dir` empty by default), so the compile-cache build cost
applies symmetrically to every sample. OS page-cache state varies
between invocations and is not controlled; rerun a few times to
gauge variance.

`WithBenchmarkEnabled` is set, and a short Generate runs after
`New` to confirm the engine loaded correctly.

The upstream `litert_lm_advanced_main` binary does not expose a flag
for this option, so no native-binary cross-check is included.

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
| `-parallel` | `true` | Parallel section loading mode. `false` calls `WithParallelSectionLoading(false)`. |
| `-prompt` | `"The capital of France is"` | Short prompt issued after `New`. |
| `-cache-dir` | (fresh temp dir) | Directory passed to `WithCacheDir`. |
| `-keep-cache` | `false` | Skip cleanup of an auto-created temp dir. |
| `-max-tokens` | `4096` | Engine total token budget. |

## Run

```sh
go run ./examples/parallel-load -model "$MODEL" -backend cpu -parallel=true
go run ./examples/parallel-load -model "$MODEL" -backend cpu -parallel=false
go run ./examples/parallel-load -model "$MODEL" -backend gpu -parallel=true
go run ./examples/parallel-load -model "$MODEL" -backend gpu -parallel=false
```

## Observations on Gemma 4 E2B (Windows 11)

Two samples per configuration, fresh cache dir each invocation:

| Backend | `-parallel` | `litertlm.New` |
|---|---|---|
| CPU | `true` (default) | 2.81 s, 2.84 s |
| CPU | `false` | 2.86 s, 2.91 s |
| GPU | `true` (default) | 3.61 s, 3.50 s |
| GPU | `false` | 2.50 s, 2.53 s |

CPU: serial is ~60 ms slower than parallel — at the noise floor for
a 2.5 GB model on this hardware. The 2.8 s baseline is dominated by
compile-cache construction, not by `.litertlm` file IO.

GPU: serial is reproducibly ~1.0 s faster than parallel on this
configuration. Cause not investigated.

## Notes

- The 2.5 GB model file caches in the OS page cache after the first
  read, so a subsequent invocation in the same boot session reads
  bytes from RAM rather than disk. Cold-with-respect-to-engine-state
  is not the same as cold-with-respect-to-OS-page-cache.
- Switching the model file invalidates both the on-disk
  compile-cache and the page cache.
