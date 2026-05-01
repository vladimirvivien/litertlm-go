# cancel — interrupt a streaming generation

Starts a long-running generation, reads the first few stream chunks,
then calls `Session.Cancel()` to ask the engine to stop. The stream
channel closes shortly after with the in-progress Final chunk so the
goroutine wrapping the foreign-thread callback unwinds cleanly.

## What this example shows

- Streaming generation via `Session.GenerateContentStreamCh` (a Go
  channel of `litertlm.StreamChunk`).
- `Session.Cancel()` issued from the *consumer* goroutine while the
  engine is still emitting tokens. Cancel signals are delivered on the
  next decode step, so a couple more chunks may arrive before Final.
- Cooperative shutdown: the consumer keeps draining the channel after
  Cancel so the producing goroutine inside `GenerateContentStreamCh`
  can finish and close it.

## Prerequisites

1. Native shared library + `libGemmaModelConstraintProvider.*` staged
   in `$LITERTLM_LIB`.
2. A `.litertlm` model file.
3. Go 1.26+.

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/cancel \
    -model /abs/path/to/gemma-4-E4B-it.litertlm
```

| Flag             | Default                                                    | Notes                                  |
| ---------------- | ---------------------------------------------------------- | -------------------------------------- |
| `-model`         | (required)                                                 | Path to a `.litertlm` model.           |
| `-prompt`        | `"Tell me a long story about a dragon and a wizard."`      | Something the model wants to keep going. |
| `-cancel-after`  | `8`                                                        | Number of stream chunks to read before cancelling. |
| `-max`           | `4096`                                                     |                                        |
| `-backend`       | `"cpu"`                                                    |                                        |
| `-lib`           | `$LITERTLM_LIB`                                            |                                        |

## Expected output

```
prompt:  Tell me a long story about a dragon and a wizard.
output:
Once upon a time, in a land

[210ms] cancelling after 8 chunks ...
 far, far away,
done (11 chunks total, wall=240ms)
```

A handful of chunks slip through between the `Cancel()` call and the
engine actually noticing — that's expected. The output stops well
before the model would have produced a full story.

## Notes

- **Cancellation is cooperative, not synchronous.** The engine checks
  the cancel flag between decode steps, so a few additional tokens may
  emit between your `Cancel()` and Final. If you need strict bounds,
  count or time-cap on the consumer side.
- **Cancel is safe with no in-flight call.** Calling `Cancel()` when
  nothing is running is a no-op, so it's fine to defer or call on
  cleanup paths.
- **Don't stop reading the channel after Cancel.** The producing
  goroutine inside `GenerateContentStreamCh` blocks on sending until
  the consumer reads — abandoning the channel would leak the
  goroutine. The example keeps draining until close.
