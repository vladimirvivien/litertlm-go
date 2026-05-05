# Example: Prefill-Decode

This example uses the lower-level `litertlm-go` API sequence of `RunPrefill` → `RunDecode`
flow instead of the all-in-one `Session.GenerateContent`. The exposes how
the model actually executes: 
- *Prefill* tokenises and processes the prompt
- *Decode* generates the response token-by-token

The `litertlm-go` low-level API mirrors the C API and allows you to measure 
the different inference phases independently (token introspection, scoring, etc.).

## What this example shows

- `Session.RunPrefill(inputs)` — feeds the prompt context into the
  session, blocks and returns when prefill is done.
- `Session.RunDecode()` — synchronously generates the response from the prefilled context
  and returns a `Responses` handle (which the caller must `Delete()`).

## Prerequisites

1. LiteRT-LM shared library files staged in`LITERTLM_LIB`.
2. A `.litertlm` model (i.e. Gemma 4). 
3. `litertlm-go`

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

