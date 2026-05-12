# Example: Per-Generation Benchmarks

Demonstrates `litertlm.WithBenchmarkEnabled()` paired with
`Response.Benchmark()` for pure-Go access to per-generation metrics
on the high-level `Client` API.

## What this example shows

- `WithBenchmarkEnabled()` at `New` time turns on benchmark
  collection in the engine.
- `client.GenerateResponse(ctx, prompt)` returns a `*Response` whose
  `Benchmark()` accessor gives a `*litertlm.Benchmark` snapshot.
- `Benchmark` is a pure-Go value: `time.Duration` for time fields,
  per-turn slices for token counts and throughput. No C handle to
  manage.

## Prerequisites

1. LiteRT-LM shared library files staged in `LITERTLM_LIB`.
2. A `.litertlm` chat-tuned model (e.g. Gemma 4).

## Run

```bash
LITERTLM_LIB=/abs/path/to/lib \
    go run ./examples/benchmarks \
    -model /abs/path/to/gemma-4-E2B-it.litertlm
```

| Flag       | Default                                                              |
| ---------- | -------------------------------------------------------------------- |
| `-model`   | (required)                                                           |
| `-prompt`  | `"The capital of France is"`                                         |
| `-backend` | `"cpu"`                                                              |
| `-turns`   | `2`                                                                  |
| `-lib`     | `$LITERTLM_LIB`                                                      |

## Expected output

```
=== Turn 1 ===
 Paris.
--- Benchmark ---
  Total init time:       1.067s
  Time to first token:   210ms
  Prefill turns: 1   Decode turns: 1
  Prefill turn 0:  304.7 tok/s  (5 tokens)
  Decode  turn 0:  22.1 tok/s   (3 tokens)

=== Turn 2 ===
 Paris.
--- Benchmark ---
  …
```

`Total init time` is the one-time engine initialisation cost and is
constant across turns. `Time to first token` typically decreases on
subsequent turns once the engine is warm.

The default prompt is completion-style: `Generate` does not apply
the chat template, so chat-tuned models can emit empty or repetitive
output for bare instructions. For chat-tuned flows that need
instructional prompts, use the [`Chat`](../chat) or
[`autotool`](../autotool) example.

## Notes

- **Without `WithBenchmarkEnabled()`, `Response.Benchmark()` returns nil.**
  Calling the accessor is safe in that state; check for a nil
  `*Benchmark` before dereferencing.
- **For low-level access**, `Session.BenchmarkInfo()` and
  `Conversation.BenchmarkInfo()` return the underlying C handle. The
  high-level `Response.Benchmark()` snapshots into Go memory and
  releases the handle for you.
