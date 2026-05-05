# Example: Candidate-Completion Scoring

This example uses the low-level `litertlm-go` API to demonstrate the fine-grained
controls that are exposed from the C-API. It prefills a prompt, then scores a candidate 
completion against it and reads the resulting log-probability + tokenized length out of the
returned `Responses`.

## What this example shows

- `Session.RunPrefill(inputs)` — establish the prompt context.
- `Session.ScoreTexts(targets, storeTokenLengths)` — score one or more
  candidates against the prefilled context. 
- `Responses.Score(i)` — extract the log-probability score for
  candidate `i`. 
- `Responses.TokenLength(i)` — extract the tokenized length when
  `storeTokenLengths=true` was passed; otherwise `(0, false)`.

## Prerequisites

1. LiteRT-LM shared library files staged in`LITERTLM_LIB`.
2. A `.litertlm` model (i.e. Gemma 4). 
3. `litertlm-go`

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

The score is a log-probability (more negative = less likely). Compare
across candidates by re-running with different `-target` values; lower
absolute magnitude means the model assigns higher probability to that
completion given the prompt.

