# Troubleshooting

Quirks and known limitations, with their root causes and fixes.

## Empty completion from a chat-tuned model

**Symptom.** `Generate` returns `""`. `NumCandidates() == 1` but
`resp.Text(0)` is empty.

**Cause.** Chat-tuned models (Gemma instruct, Llama-Instruct, …) end
their reply with an end-of-sequence token immediately when given a
bare instruction prompt that hasn't been wrapped in the model's chat
template.

**Fix.** Either:

- Use a *completion-style* prompt the model can extend. The default
  `"The capital of France is"` works on Gemma 4 and many others.
- Use the high-level [`Chat`](chat.md) API or the low-level
  `Conversation` API — both apply the model's chat template
  automatically.

---

## Detokenized output contains `▁` (U+2581) instead of spaces

**Symptom.** `Engine.Detokenize` returns text with the lower-one-eighth-
block character `▁` where you'd expect ASCII spaces:

```
"Hello,▁world.▁How▁are▁you?"
```

**Cause.** That's SentencePiece's internal space marker. The C API
faithfully passes through the tokenizer's raw output without
post-processing it.

**Fix.** Fix in Go if you want plain spaces:

```go
out = strings.ReplaceAll(out, "▁", " ")
```
---

## Empty `default.profraw` files appear in working directory

**Symptom.** Running anything that loads the LiteRT-LM library leaves
a zero-byte `default.profraw` file in the current working directory.

**Cause.** The prebuilt LiteRT-LM dependencies under
`prebuilt/<os>/lib*.so` were compiled with LLVM
`-fprofile-instr-generate`. The embedded `__llvm_profile_*` runtime
writes a coverage dump on process exit to `./default.profraw`.

**Fix.** Set `LLVM_PROFILE_FILE` to a discardable target before
running:

=== "Linux/macOS"

    ```bash
    LLVM_PROFILE_FILE=/dev/null \
    LITERTLM_LIB=/path/to/lib \
        go run main.go
    ```

=== "Windows"

    ```powershell
    $Env:LLVM_PROFILE_FILE = "NUL"
    $Env:LITERTLM_LIB = "C:\path\to\lib"
    go run main.go
    ```
---

## `engine_create` returns nil with `DYNAMIC_UPDATE_SLICE` errors

**Symptom.** `litertlm.New` (or low-level `NewEngine`) errors during
construction; the C-side log mentions `DYNAMIC_UPDATE_SLICE`.

**Cause.** `WithMaxTokens(n)` is set below the model's smallest
prefill signature (typically 128).

**Fix.** Use `WithMaxTokens(1024)` or higher. The high-level `Client`
defaults to 4096, which works for every Gemma 4 variant.

---

## Panic: `litertlm: missing C symbol "..." in loaded library`

**Symptom.** A call into the Go API panics with a message like:

```
panic: litertlm: missing C symbol "litert_lm_session_config_set_apply_prompt_template"
in loaded library (refresh the prebuilt LiteRT-LM libs to a build that
exports it): symbol not found
```

`litertlm.Load` succeeds — the panic fires the first time a method
whose underlying C symbol is missing actually runs (which may be
during `litertlm.New`, or later when you invoke a specific feature).

**Cause.** The Go bindings resolve C symbols lazily, on first call.
The LiteRT-LM library staged in `$LITERTLM_LIB` predates the named
symbol — typically because the prebuilt libs were copied from an
older upstream build than this `litertlm-go` release was compiled
against.

**Fix.** Re-stage the prebuilt LiteRT-LM libraries from a current
upstream build per
[`LITERTLM-BUILD.md`](https://github.com/vladimirvivien/litertlm-go/blob/main/LITERTLM-BUILD.md).

---

## GPU run falls back to CPU

**Symptom.** Running with `WithBackend("gpu")` logs
`WARNING: GPU accelerator could not be loaded and registered` and
inference continues on CPU.

**Cause.** One or more GPU plugins
(`libLiteRtWebGpuAccelerator.*`, `libLiteRtTopKWebGpuSampler.*`, the
DirectX Shader Compiler on Windows) are missing from `$LITERTLM_LIB`.

**Fix.** Re-stage the prebuilt accelerator plugins per
[`LITERTLM-BUILD.md` §4](https://github.com/vladimirvivien/litertlm-go/blob/main/LITERTLM-BUILD.md)
(or the Windows equivalent).

---

## `tool_calls.arguments` numeric values come as `float64`

**Symptom.**
`reply.ToolCalls()[0].Function.Arguments["count"].(int)` panics with
"interface conversion: interface {} is float64, not int".

**Cause.** `encoding/json` decodes JSON numbers into `float64` by
default. The Go `Arguments` map is `map[string]any`, and the model
emits all numbers as JSON numbers regardless of the schema you
declared.

**Fix.** Type-assert to `float64` and convert if needed:

```go
n := int(reply.ToolCalls()[0].Function.Arguments["count"].(float64))
```

---

## Markers `<|"|>` in tool-call argument values

**Symptom.** A tool-call argument has the literal text
`<|"|>Boston, MA<|"|>` instead of `Boston, MA`.

**Cause.** Gemma 4's chat-template renderer leaves its internal quote
markers in string-typed arguments when the C side surfaces them as
JSON.

**Fix.** The high-level `*Reply` strips these automatically on
parse, so callers see clean values. If you're working at the low
level (`Conversation.SendMessage`), you will need to strip them yourself:

```go
strings.ReplaceAll(arg, `<|"|>`, "")
```
