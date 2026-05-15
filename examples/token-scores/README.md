# token-scores

Score one or more candidate continuations against a prefilled prompt
and print the per-token log-probability scores alongside the
detokenized text.

## What this exercises

| Surface | Use |
|---|---|
| `Session.ScoreTexts(targets, storeTokenLengths)` | Score a candidate against the current prefill state. One target per call on the CPU engine (see Constraints). |
| `Responses.Score(i)` | Candidate-level score. |
| `Responses.TokenScores(i)` | Per-token scores for candidate `i`. |
| `Engine.Tokenize(text)` | Resolve a candidate string to its token ids so each per-token score has a label. |
| `Engine.Detokenize(ids)` | Decode each id back to readable text for the score table. |

## Constraints

- The CPU engine rejects `num_targets > 1` in a single `ScoreTexts`
  call. Two candidates means two `Session`s against the same
  `Engine`, each with its own prefill of the same prompt.
- Score sign convention: `ScoreTexts` returns log-probabilities.
  Higher (less negative) means more likely; total score is the sum
  across tokens.
- Subword tokenization: SentencePiece encodes leading spaces as the
  `▁` marker. A candidate like `" Paris"` may resolve to a single
  token (`▁Paris`) on Gemma 4.

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
| `-prompt` | `"The capital of France is"` | Prefilled context. |
| `-candidates` | `" Paris, London"` | Comma-separated continuations. Each is scored in its own Session. Leading space is restored after splitting so SentencePiece's `▁` marker fires. |
| `-max-tokens` | `2048` | Engine total token budget. |

## Run

```sh
go run ./examples/token-scores -model "$MODEL" -backend cpu
go run ./examples/token-scores -model "$MODEL" -backend cpu \
    -candidates "Paris and the surrounding area, Lyon and the countryside"
```

## Observed output

Gemma 4 E2B, CPU, default candidates:

```
prompt: "The capital of France is"

candidate:   " Paris"
total score: -0.0074
  token_id   token        score
  9079       "▁Paris"     -0.0074

candidate:   " London"
total score: -13.6199
  token_id   token        score
  5860       "▁London"    -13.6199
```

The Paris log-prob is essentially zero — the model is near-certain.
London is ~13.6 nats less likely, i.e. roughly `e^-13.6 ≈ 1.2e-6`
relative probability.

With multi-token candidates:

```
candidate:   " Paris and the surrounding area"
total score: -18.8340
  token_id   token        score
  9079       "▁Paris"     -0.0074
  532        "▁and"       -6.1434
  506        "▁the"       -0.5368
  12989      "▁surrounding" -10.4213
  2433       "▁area"      -1.7250

candidate:   " Lyon and the countryside"
total score: -32.5421
  token_id   token        score
  49533      "▁Lyon"      -10.8219
  532        "▁and"       -6.5141
  506        "▁the"       -0.2533
  34098      "▁countryside" -14.9528
```

The first-token gap (`-0.0074` for Paris vs `-10.8` for Lyon)
dominates the totals; the conditional tokens (`▁and`, `▁the`) score
similarly across both candidates.

## Notes

- Per-token slice length matches `len(ids)` from `Engine.Tokenize`
  on the candidate text. A `-` placeholder appears in the score
  column if the score slice is shorter than the id slice
  (defensive; not expected on the CPU engine path).
- For the candidate-level use case without per-token detail, see
  `examples/score/`.
