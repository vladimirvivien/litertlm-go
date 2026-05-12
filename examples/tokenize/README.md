# Example: Text Tokenization

Tokenizes a string with `Engine.Tokenize` and detokenizes the result
back to bytes — the low-level path for non-generation engine features.

## What this example shows

- `litertlm.Load` to load the runtime library for a chosen backend.
- `Engine` construction via `EngineSettings`.
- `Engine.Tokenize(text)` returning the token-ID slice.
- `Engine.Detokenize(ids)` round-tripping back to bytes.

## Prerequisites

1. LiteRT-LM shared library files staged in `LITERTLM_LIB`.
2. A `.litertlm` model (e.g. Gemma 4).


## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/tokenize \
    -model /abs/path/to/gemma-4-E4B-it.litertlm
    -model "Hello, world. How are you today?"
```

| Flag       | Default                                     | Notes                                          |
| ---------- | ------------------------------------------- | ---------------------------------------------- |
| `-model`   | (required)                                  | Path to a `.litertlm` model.                   |
| `-text`    | `"Hello, world. How are you today?"`        | The string to tokenize.                        |
| `-backend` | `"cpu"`                                     |                                                |
| `-lib`     | `$LITERTLM_LIB`                             |                                                |

## Expected output

```
text:        "Hello, world. How are you today?"
tokens (9):  [9259 236764 1902 236761 2088 659 611 3124 236881]
round-trip:  "Hello,▁world.▁How▁are▁you▁today?"
```

Token ids are model-specific — different `.litertlm` files produce
different splits.

The `▁` character (U+2581) in the detokenized output is
SentencePiece's internal space marker.
