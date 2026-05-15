# benchmarks

Contrast the two per-generation benchmark surfaces against the same
`Client` and `Engine`:

| Path | Producer | Accessor | Return |
|---|---|---|---|
| High-level | `Client.GenerateResponse` | `Response.Benchmark()` | `*litertlm.Benchmark` — pure-Go snapshot; no handle to manage. |
| Low-level | `Session.GenerateContent` | `Session.BenchmarkInfo()` | `litertlm.BenchmarkInfo` — C-side handle; caller `Delete()`s it. |

Both require `WithBenchmarkEnabled()` at `New` time. Without it,
`Response.Benchmark()` returns `nil` and the C-side `BenchmarkInfo`
fields read as zero.

## Prerequisites

- `.litertlm` chat-tuned model (e.g. Gemma 4).
- LiteRT-LM shared libraries on disk. Pass `-lib <dir>` or set
  `LITERTLM_LIB=<dir>`.

## Flags

| Flag | Default | Description |
|---|---|---|
| `-model` | (required) | Path to `.litertlm` model file. |
| `-lib` | `$LITERTLM_LIB` | Directory holding the shared libraries. |
| `-backend` | `cpu` | `cpu` or `gpu`. |
| `-prompt` | `"The capital of France is"` | Completion-style prompt; `Generate` does not apply the chat template. |
| `-turns` | `2` | Number of high-level Generate calls before the low-level section. |

## Run

```sh
go run ./examples/benchmarks -model "$MODEL"
```

## Observed output

Gemma 4 E2B, CPU, `-turns 1`:

```
=== High-level turn 1 (Response.Benchmark) ===
 Paris.
--- *Benchmark (Go) ---
  Total init time:     517.7532ms
  Time to first token: 465.361966ms
  Prefill turns: 1   Decode turns: 1
  Prefill turn 0: 14.7 tok/s (6 tokens)
  Decode  turn 0: 17.5 tok/s (3 tokens)

=== Low-level (Session.BenchmarkInfo) ===
 Paris.
--- BenchmarkInfo (C handle) ---
  Time to first token: 0.403s
  Prefill turns: 1   Decode turns: 1
  Prefill turn 0: 16.6 tok/s (6 tokens)
  Decode  turn 0: 24.2 tok/s (3 tokens)
```

`Total init time` is the one-time engine initialisation cost; the
C-side `BenchmarkInfo` exposes it via separate accessors not
printed here.

The two paths share an `Engine` (the low-level section uses
`client.Engine().NewSession(0)`), so the model file is loaded once.
Per-generation throughput differs across the two runs by ordinary
run-to-run noise — both numbers reflect warmed engine state.

## Notes

- `Response.Benchmark()` snapshots into Go memory and releases the
  underlying handle. The returned `*Benchmark` survives after
  `Response` is collected.
- `Session.BenchmarkInfo()` returns a `BenchmarkInfo` C handle the
  caller must `Delete()`. Reading after `Delete()` is undefined.
- The high-level `Generate` path opens and deletes a Session per
  call, so each `Response.Benchmark()` captures one prefill + one
  decode turn. A long-lived `Session` (low-level) accumulates turn
  history across calls.
