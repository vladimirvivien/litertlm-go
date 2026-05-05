# Example: Introduction to the litertlm-go

This is a simple "hello world" introduction to  `litertlm-go`.

## What this example shows

- Creates a new instance of `litertlmgo.Client`
- Use `Client.Generate` to do a one-shot text generate from the LLM.
- Prints the textual reply 



## Prerequisites

1. LiteRT-LM shared library files staged in`LITERTLM_LIB`.
2. A `.litertlm` model (i.e. Gemma 4). 
3. `litertlm-go`

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
| `-prompt`   | `"The capital of France is"`         | The text fed to the model. Pick a *completion-style* prompt.   |
| `-backend`  | `"cpu"`                              | Set to `"gpu"` if you staged the GPU-capable build (see `gpu` example). |
| `-max`      | `1024`                               | Total token budget. Must be ≥ the model's smallest prefill signature (typically 128). |
| `-lib`      | `$LITERTLM_LIB`                      | Override the lib directory without touching the env var.       |

