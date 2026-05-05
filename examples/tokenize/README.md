# Example: Text Tokenization

This example uses the lower-level of `litertlm-go` API to demostrate 
access to the engines non-generation features. It implements a demo of 
the engine's tokenizer.

## What this example shows
- Use `litertlm.Load` to load the runtime library files for the specified backend.
- Configure `Engine` using `EngineSettings`
- Then use `Engine.Tokenize` to generate token IDs.
- Use `Engine.Detokenize` to reverse the process and produce equivalent bytes.

## Prerequisites

1. LiteRT-LM shared library files staged in`LITERTLM_LIB`.
2. A `.litertlm` model (i.e. Gemma 4). 
3. `litertlm-go`


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

The `▁` character (U+2581 LOWER ONE EIGHTH BLOCK) in the detokenized
output is **expected** — it's SentencePiece's internal space marker.
