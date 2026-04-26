# hello — minimal synchronous inference

A "hello world" for litertlm-go. Loads a model, opens a session, runs one
synchronous `GenerateContent` call against the **low-level Session API**,
and prints every candidate the model returns.

## What this example shows

- The minimum lifecycle: `Load` → `EngineSettings` → `Engine` → `Session`
  → `GenerateContent` → `Responses`.
- Correct cleanup with `defer ...Delete()` in reverse-creation order.
- How to feed a UTF-8 prompt as `InputData` of type `InputText`.
- A defensive check for the empty-completion gotcha that catches
  first-time users (see [Notes](#notes)).

## Prerequisites

1. **Native shared library + runtime deps staged in a directory**, per
   `LITERTLM-BUILD.md` §3 and §4. At minimum:
   - `liblitertlm_c_cpu.so` (Linux) / `.dylib` (macOS) / `.dll` (Windows)
   - `libGemmaModelConstraintProvider.{so,dylib,dll}`
2. **A `.litertlm` model file**. Gemma 4 2B works well for a smoke test:
   - `litert-lm pull --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm`
3. **Go 1.22 or newer** (`go version`).

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

## Expected output

A single candidate with the model's continuation:

```
[0] Paris. It is the largest city in France and the country's political, economic, and cultural center.
```

Total wall time on a recent laptop CPU with Gemma 4 2B: ~10–20 s for the
first run (engine init dominates), a few seconds per subsequent prompt
within the same process.

## Notes

- **Empty completion?** This example uses the raw `Session.GenerateContent`
  call, which sends your prompt to the model **as plain text without any
  chat template wrapping**. Chat-tuned models (Gemma instruct, Llama-Instruct,
  …) often respond to a bare instruction-style prompt by emitting an
  end-of-sequence token immediately, returning an empty candidate. The
  example detects this and prints a hint pointing you at the [`chat`](../chat/)
  example, which uses the Conversation API and applies the model's chat
  template automatically.

- **Why `-max 1024`?** A smaller value (`-max 64`, etc.) trips the engine
  with `DYNAMIC_UPDATE_SLICE` errors because the model's smallest prefill
  signature can't fit. Stick with the default unless you know the model's
  signatures.

- This example deliberately avoids streaming so you can see the bare
  request/response shape. For incremental output, see the
  [`stream`](../stream/) example.
