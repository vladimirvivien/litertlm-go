# Example: Hello World

Minimal one-shot generation against a `.litertlm` model.

## What this example shows

- `litertlm.New(ctx, ...)` to construct a `Client`.
- `Client.Generate(ctx, prompt)` for a one-shot text completion.
- Printing the returned string.

## Prerequisites

1. LiteRT-LM shared library files staged in `LITERTLM_LIB`.
2. A `.litertlm` model (e.g. Gemma 4).

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/hello \
    -model /abs/path/to/gemma-4-E2B-it.litertlm
```

Optional flags:

| Flag        | Default                              | Notes                                                          |
| ----------- | ------------------------------------ | -------------------------------------------------------------- |
| `-model`    | (required)                           | Path to the `.litertlm` file.                                  |
| `-prompt`   | `"The capital of France is"`         | Completion-style prompt; `Generate` does not apply the chat template. |
| `-backend`  | `"cpu"`                              | Set to `"gpu"` if you staged the GPU-capable build (see `gpu` example). |
| `-max`      | `1024`                               | Total token budget. Must be ≥ the model's smallest prefill signature (typically 128). |
| `-lib`      | `$LITERTLM_LIB`                      | Override the lib directory without touching the env var.       |
| `-loglevel` | `quiet`                              | LiteRT-LM log severity floor. Accepts `verbose` / `debug` / `info` / `warning` / `error` / `fatal` / `quiet` (also numeric). Pass `info` to see the C-side loader chatter. |

