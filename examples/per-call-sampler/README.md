# per-call-sampler

Override the sampler shape per `Generate` call against a single
`Client`. Three calls, three samplers, one prompt — the divergence
between replies is the demonstration.

## What this exercises

| Surface | Use |
|---|---|
| `WithSampler(p SamplerParams)` | Per-call sampler override. |
| `WithMaxOutputTokens(n)` | Per-call decode cap, shared across all three calls so replies are comparable in length. |

## Sampler shapes

| Label | Params | Behavior |
|---|---|---|
| Deterministic | `Type: SamplerTopP, TopP: 0.95, TopK: 40, Temperature: 0` | Argmax. `Temperature: 0` collapses softmax to a single candidate; re-runs produce the identical reply. |
| Balanced | `Type: SamplerTopP, TopP: 0.9, TopK: 40, Temperature: 0.7` | Mild variability across runs. |
| Creative | `Type: SamplerTopP, TopP: 0.95, TopK: 40, Temperature: 1.2` | High variability; can produce off-distribution tokens. |

All three use `SamplerTopP`. As of LiteRT-LM v0.x, only
`SamplerTopP` (value 2) is implemented at the C side; passing
`SamplerTopK` (1) or `SamplerGreedy` (3) fails with
`UNIMPLEMENTED: Sampler type: N not implemented yet` from
`engine_create_session`. Use `Temperature: 0` with `SamplerTopP`
when deterministic output is needed.

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
| `-prompt` | `"Write a haiku about the changing seasons."` | Prompt issued for all three calls. |
| `-seed` | `1` | Sampler seed; affects the variable shapes (TopP > 0 temperature). |
| `-max-tokens` | `96` | Per-call decode cap. |

## Run

```sh
go run ./examples/per-call-sampler -model "$MODEL" -backend cpu
```

## Observed output

Gemma 4 E2B, CPU, Windows 11, default flags:

```
prompt: Write a haiku about the changing seasons.

=== Run 1: Deterministic (TopP=0.95, Temp=0) ===

Green leaves turn to gold,
Winter whispers on the breeze,
Spring blooms softly now.

=== Run 2: Balanced (TopP=0.9, Temp=0.7) ===

Green leaves turn to gold,
Cold winds whisper through the trees,
New life sleeps beneath.

=== Run 3: Creative (TopP=0.95, Temp=1.2) ===
***
Four green leaves turn gold now,
A gentle breeze whispers home,
Four new life sleeps soon. आइसित्
```

Re-running with the same seed reproduces the Deterministic cell
verbatim. The Balanced and Creative cells differ across runs even
with the same seed; the Creative sampler occasionally emits
off-distribution tokens (e.g. the Devanagari fragment at the end of
Run 3) as a side effect of the high temperature exploring tail
probabilities.

## Notes

- `WithSampler` overrides any `WithDefaultSampler` set on the
  Client for the duration of the single call.
- The shared `WithMaxOutputTokens` keeps the three replies short
  enough to compare side-by-side. Increase `-max-tokens` to see
  longer divergence.
- Effects on quality are model-dependent. Temperatures above ~1.0
  on small models will frequently drift into off-distribution
  tokens, which is the point of the Creative cell.
