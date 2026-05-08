# Example: Speculative Decoding Comparison

Compares decode throughput with and without
`litertlm.WithSpeculativeDecodingEnabled(true)`. The example loads the
model twice — speculative decoding is configured at engine-creation
time, not per call — runs the same prompt against each Client, and
prints a side-by-side benchmark plus the computed speedup ratio.

Speculative (multi-token-prediction) decoding lets the model emit
several candidate tokens per forward pass and accept whichever
agree with the verification step. On models that support it
(e.g. Gemma 4) this typically yields 1.5–3× decode-throughput
gains; the actual speedup varies with prompt length, hardware,
and draft acceptance rate.

## What this example shows

- Toggling speculative decoding via `WithSpeculativeDecodingEnabled`
  on a `litertlm.Client`.
- Capturing per-call benchmarks via
  `WithBenchmarkEnabled` + `Response.Benchmark()`.
- Computing decode-throughput and wall-clock speedup ratios from the
  two `*Benchmark` snapshots.

## Prerequisites

1. LiteRT-LM shared library files staged in `LITERTLM_LIB`. The
   library must be built with speculative-decoding support; if not,
   the second `litertlm.New(...)` call will surface a clear error.
2. A `.litertlm` model that supports multi-token prediction —
   Gemma 4 is the reference target.

## Run

```bash
LITERTLM_LIB=/abs/path/to/lib \
    go run ./examples/speculative \
    -model /abs/path/to/gemma-4-E4B-it.litertlm
```

| Flag           | Default                                                       | Notes                                            |
| -------------- | ------------------------------------------------------------- | ------------------------------------------------ |
| `-model`       | (required)                                                    | Path to a speculative-decoding-capable model.    |
| `-prompt`      | `"The capital of France is"`                                  | Completion-style; works for chat-tuned models.   |
| `-backend`     | `"cpu"`                                                       | `gpu` if you staged the GPU build.               |
| `-max-output`  | `256`                                                         | Decode budget per call.                          |
| `-lib`         | `$LITERTLM_LIB`                                               |                                                  |

## Expected output

```
=== Without speculative decoding ===
 Paris.
  Init time:               1.07s
  Time to first token:     312ms
  Wall-clock generate:     3.55s
  Prefill turn 0:          304.7 tok/s  (5 tokens)
  Decode  turn 0:          36.1 tok/s   (128 tokens)

=== With speculative decoding (multi-token prediction) ===
 Paris.
  Init time:               1.21s
  Time to first token:     298ms
  Wall-clock generate:     1.77s
  Prefill turn 0:          307.0 tok/s  (5 tokens)
  Decode  turn 0:          72.4 tok/s   (128 tokens)

=== Speedup ===
  Decode tokens (off / on):  128 / 128
  Decode tok/s  (off / on):  36.1 / 72.4   (2.00×)
  Wall-clock    (off / on):  3.55s / 1.77s   (2.01×)
```

Numbers above are illustrative; actual values depend on hardware,
prompt length, and the model's draft-token acceptance rate.

## Notes

- **Two engine loads.** Because speculative decoding is engine-scoped,
  the example builds two Clients in sequence. Total wall time
  roughly doubles compared to a single inference run. For a faster
  smoke test, use the smaller E2B model.
- **Speedup is workload-dependent.** Short replies (a few tokens)
  may show no measurable speedup or even a small regression because
  the speculative-decoding overhead isn't amortised. The benefit
  scales with decode length.
- **Library compatibility.** If your LiteRT-LM build doesn't support
  multi-token prediction, the speculative `New(...)` call returns a
  clear engine-construction error. The first (baseline) Client
  loads independently and still works.
