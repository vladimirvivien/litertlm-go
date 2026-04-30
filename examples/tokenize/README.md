# tokenize — text ↔ token-id round-trip

A small offline demo of the engine's tokenizer. Loads a model, runs no
inference, and shows what `Engine.Tokenize` and `Engine.Detokenize`
return for a sample string.

## What this example shows

- The minimum lifecycle for tokenizer-only access:
  `Load` → `EngineSettings` → `Engine` → `Tokenize` / `Detokenize`.
- That `Engine.Tokenize` returns a `[]int32` of model-native token ids,
  not pieces or strings.
- That `Engine.Detokenize` reverses the operation and produces a
  byte-equivalent string for normal inputs.

## Prerequisites

1. Native shared library + `libGemmaModelConstraintProvider.*` staged
   in a directory pointed to by `LITERTLM_LIB`.
2. A `.litertlm` model file (any chat-tuned Gemma 4 works).
3. Go 1.26+.

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/tokenize \
    -model /abs/path/to/gemma-4-E4B-it.litertlm
```

Override the input with `-text`:

```bash
go run ./examples/tokenize -model … -text "The quick brown fox."
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
LiteRT-LM's C API surfaces the tokenizer's raw output without
post-processing it back to ASCII spaces. If your application needs
plain spaces, normalise on the Go side:

```go
out = strings.ReplaceAll(out, "▁", " ")
```

## Notes

- The example deliberately avoids `NewSession`/`GenerateContent`. The
  tokenizer lives on the Engine handle and is available the moment the
  engine is constructed.
- `Engine.Tokenize` copies the C-side token array into Go memory before
  releasing the underlying `LiteRtLmTokenizeResult`, so the returned
  slice stays valid after the call returns.
- For models that include special tokens (BOS / EOS / system markers),
  `Tokenize` does NOT inject them automatically — pass them in the
  input text or rely on the chat template via the `conversation`
  example for full prompt construction.
