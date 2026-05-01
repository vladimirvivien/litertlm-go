# prefill-decode — explicit two-phase generation

Drives the engine through the lower-level `RunPrefill` → `RunDecode`
flow instead of the all-in-one `Session.GenerateContent`. The split
mirrors how the model actually executes — *prefill* tokenises and
processes the prompt; *decode* generates the response token-by-token —
and lets you measure the two phases independently or interleave them
with other operations (token introspection, scoring, etc.).

## What this example shows

- `Session.RunPrefill(inputs)` — feeds the prompt context into the
  session. Blocking; returns when prefill is done.
- `Session.RunDecode()` — generates from the prefilled context.
  Blocking; returns a `Responses` handle the caller must `Delete()`.
- Per-phase wall-clock timing so you can see the prefill/decode split.

## Prerequisites

1. Native shared library + `libGemmaModelConstraintProvider.*` staged
   in `$LITERTLM_LIB`.
2. A `.litertlm` model file (Gemma 4 instruct works).
3. Go 1.26+.

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/prefill-decode \
    -model /abs/path/to/gemma-4-E4B-it.litertlm
```

| Flag       | Default                             | Notes                                     |
| ---------- | ----------------------------------- | ----------------------------------------- |
| `-model`   | (required)                          | Path to a `.litertlm` model.              |
| `-prompt`  | `"The capital of France is"`        | Completion-style prompt.                  |
| `-max`     | `2048`                              | Total token budget (prompt + output).     |
| `-backend` | `"cpu"`                             |                                           |
| `-lib`     | `$LITERTLM_LIB`                     |                                           |

## Expected output

```
prefill:  120ms   ("The capital of France is")
decode:   2.3s
response:  the capital of France is the capital of France.

**The capital of France is Paris.**
```

Times vary with hardware; prefill is roughly linear in prompt length
and decode is roughly linear in output length.

## Notes

- **Pair RunPrefill/RunDecode on a fresh session.** The C engine
  enforces a single prefill→decode cycle per session — re-running
  prefill on the same session after a decode/score errors out with
  `new_step must be less than or equal to TokenCount()`. Open a new
  `Session` for each independent generation.
- **For multi-turn chat, use Conversation API** (see the
  [`conversation`](../conversation/) example), which manages prefill
  state across turns internally.
- **`GenerateContent` is the higher-level shortcut** — it does
  prefill + decode + final-token cleanup in one call. Prefer it unless
  you specifically want the split.
