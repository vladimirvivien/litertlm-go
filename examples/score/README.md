# Example: Candidate-Completion Scoring

Prefills a prompt, scores a candidate completion against it, and
reads the log-probability and tokenized length from the returned
`Responses` handle.

## What this example shows

- `Session.RunPrefill(inputs)` — establish the prompt context.
- `Session.ScoreTexts(targets, storeTokenLengths)` — score one or more
  candidates against the prefilled context.
- `Responses.Score(i)` — log-probability score for candidate `i`.
- `Responses.TokenLength(i)` — tokenized length when
  `storeTokenLengths=true` was passed; otherwise `(0, false)`.

## Prerequisites

1. LiteRT-LM shared library files staged in `LITERTLM_LIB`.
2. A `.litertlm` model (e.g. Gemma 4).

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/score \
    -model /abs/path/to/gemma-4-E4B-it.litertlm
```

Override the prompt or candidate:

```bash
go run ./examples/score -model … \
    -prompt "The capital of Spain is" \
    -target " Madrid."
```

| Flag       | Default                             | Notes                                          |
| ---------- | ----------------------------------- | ---------------------------------------------- |
| `-model`   | (required)                          | Path to a `.litertlm` model.                   |
| `-prompt`  | `"The capital of France is"`        | Prefilled context.                             |
| `-target`  | `" Paris."`                         | Single candidate completion (engine rejects more). |
| `-max`     | `2048`                              |                                                |
| `-backend` | `"cpu"`                             |                                                |
| `-lib`     | `$LITERTLM_LIB`                     |                                                |

## Expected output

```
prompt:    "The capital of France is"
candidate: " Paris."
[0] text=" Paris." score=-6.558003 (ok=true) tokenLen=2 (ok=true)
```

The score is a log-probability: lower absolute magnitude indicates
higher model probability for that completion. Re-run with different
`-target` values to compare candidates.
