# score — candidate-completion scoring

Prefills a prompt, then scores a candidate completion against it and
reads the resulting log-probability + tokenized length out of the
returned `Responses`.

## What this example shows

- `Session.RunPrefill(inputs)` — establish the prompt context.
- `Session.ScoreTexts(targets, storeTokenLengths)` — score one or more
  candidates against the prefilled context. The CPU engine currently
  refuses `len(targets) > 1` (`INVALID_ARGUMENT: Target text size
  should be 1`), so this example sends a single candidate.
- `Responses.Score(i)` — extract the log-probability score for
  candidate `i`. Returns `(value, ok)` mirroring the C
  `has_score_at` / `get_score_at` pair.
- `Responses.TokenLength(i)` — extract the tokenized length when
  `storeTokenLengths=true` was passed; otherwise `(0, false)`.

## Prerequisites

1. Native shared library + `libGemmaModelConstraintProvider.*` staged
   in `$LITERTLM_LIB`.
2. A `.litertlm` model file.
3. Go 1.26+.

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

## Notes

- **`Score`'s `ok` is mostly informational.** The C `has_score_at`
  predicate returns `true` whenever the index is in range, *not* only
  when a real score was computed. `Responses` from `GenerateContent`
  or `RunDecode` will return `(0, true)` — placeholder, not a real
  zero score. Use `ScoreTexts` if you need meaningful values.
- **One candidate at a time.** Run scoring multiple times if you want
  to compare candidates; each run can stay on the same session as long
  as you call `Cancel()` or open a fresh session between them — the
  engine doesn't allow re-prefill on a session that's already decoded
  or scored.
- **`storeTokenLengths`.** Pass `true` if you intend to read
  `Responses.TokenLength`; otherwise it returns `(0, false)`.
