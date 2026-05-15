# tokenize

Tokenize a string and inspect the model's start / stop token
metadata. No inference is run.

## What this exercises

| Surface | Use |
|---|---|
| `Client.Tokenize(text)` | High-level shortcut returning `[]int32`. Same result as `Client.Engine().Tokenize(text)`. |
| `Client.TokenLength(text)` | Token count only — useful for budget bookkeeping without allocating the id slice. |
| `Client.Engine().Detokenize(ids)` | Reverse direction, ids back to text. |
| `Client.Engine().StartTokenIDs()` | Model's configured BOS token, when present. |
| `Client.Engine().StopTokenIDs()` | Model's configured EOS token sequences. |

`Client.Engine()` returns the underlying `Engine` handle. Lifetime
is the Client's; do not call `Engine.Delete()` on it.

## Prerequisites

- `.litertlm` model file.
- LiteRT-LM shared libraries on disk. Pass `-lib <dir>` or set
  `LITERTLM_LIB=<dir>`.

## Flags

| Flag | Default | Description |
|---|---|---|
| `-model` | (required) | Path to `.litertlm` model file. |
| `-lib` | `$LITERTLM_LIB` | Directory holding the shared libraries. |
| `-backend` | `cpu` | `cpu` or `gpu`. |
| `-text` | `"Hello, world. How are you today?"` | Text to tokenize. |

## Run

```sh
go run ./examples/tokenize -model "$MODEL"
```

## Observed output

Gemma 4 E2B:

```
text:         "Hello, world. How are you today?"
Client.Tokenize    (9): [9259 236764 1902 236761 2088 659 611 3124 236881]
Client.TokenLength (9)
Engine.Detokenize:    "Hello,▁world.▁How▁are▁you▁today?"
Engine.StartTokenIDs: [2]
Engine.StopTokenIDs:  [[1] [50] [106]]
```

Token ids are model-specific. The `▁` character (U+2581) is
SentencePiece's internal space marker.
