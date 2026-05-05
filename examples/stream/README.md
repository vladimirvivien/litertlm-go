# Example: Token-by-Token Streaming

This example uses `litertlm.GenerateStream()` to the demostrate the streaming
capability of the engine.

## What this example shows

- Creates and configures a new `litertlm.Client`.
- Uses `litertlm.GenerateStream()` to request a one-shot generation from the model.
- Uses `for-range` loop to process each `litertlm.StreamChunk` until it encounters
  `litertlm.Final` is true.

## Prerequisites

1. LiteRT-LM shared library files staged in`LITERTLM_LIB`.
2. A `.litertlm` model (i.e. Gemma 4). 
3. `litertlm-go`

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/stream \
    -model /abs/path/to/gemma-4-E2B-it.litertlm \
    -prompt "Write a short haiku about the sea."
```

| Flag        | Default                                   |
| ----------- | ----------------------------------------- |
| `-model`    | (required)                                |
| `-prompt`   | `"Write a short haiku about the sea."`    |
| `-backend`  | `"cpu"`                                   |
| `-max`      | `1024`                                    |
| `-lib`      | `$LITERTLM_LIB`                           |

## Expected output

The text appears chunk-by-chunk, terminated by a newline once `Final=true`:

```
Blue waves crash on shore,
Salt wind whispers secrets deep,
Ocean calls to soul.
```