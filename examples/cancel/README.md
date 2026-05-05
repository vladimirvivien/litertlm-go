# Example: Cancel/Interrupt Streaming Generation

This examples shows how to use the `litertlm-go` API to cancel  a 
long-running response generation.

## Prerequisites

1. LiteRT-LM shared library files staged in`LITERTLM_LIB`.
2. A `.litertlm` model (i.e. Gemma 4). 
3. `litertlm-go`

## What this example shows

- Create a new litertlm-go `Client`
- Create a cancellable `context.Context`
- Use the context to initiate a stream geneartion via `Client.GenerateStream`.
- Call `Context.CancelFn()` after receiving `n` count of message `litertlm.Chunk`

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
