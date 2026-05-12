# Example: Prefill-Decode

Runs the two-phase generation flow (`RunPrefill` → `RunDecode`)
explicitly instead of the all-in-one `Session.GenerateContent`.

- *Prefill* tokenises and processes the prompt.
- *Decode* generates the response token-by-token.

The low-level API mirrors the C API and exposes each phase for
independent timing or interleaving with other engine calls.

## What this example shows

- `Session.RunPrefill(inputs)` — feeds prompt context into the
  session and blocks until prefill completes.
- `Session.RunDecode()` — synchronously generates the response from
  the prefilled context and returns a `Responses` handle (caller
  must `Delete()`).

## Prerequisites

1. LiteRT-LM shared library files staged in `LITERTLM_LIB`.
2. A `.litertlm` model (e.g. Gemma 4).

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

Prefill duration scales with prompt length; decode duration scales
with output length.
