# Example: Cancel/Interrupt Streaming Generation

Cancels a streaming generation mid-flight by cancelling the
`context.Context` passed to `Client.GenerateStream`.

## Prerequisites

1. LiteRT-LM shared library files staged in `LITERTLM_LIB`.
2. A `.litertlm` model (e.g. Gemma 4).

## What this example shows

- `Client.GenerateStream(ctx, prompt)` returning an
  `iter.Seq2[Chunk, error]`.
- Cancelling the context after N received chunks; the iterator
  observes the cancellation and terminates.

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/cancel \
    -model /abs/path/to/gemma-4-E4B-it.litertlm
```

| Flag             | Default                                                    | Notes                                  |
| ---------------- | ---------------------------------------------------------- | -------------------------------------- |
| `-model`         | (required)                                                 | Path to a `.litertlm` model.           |
| `-prompt`        | `"Tell me a long story about a dragon and a wizard."`      | A long-form prompt so cancellation lands mid-decode. |
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
