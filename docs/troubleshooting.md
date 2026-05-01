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

**Fix.** Normalise on the Go side if you want plain spaces:

```go
out = strings.ReplaceAll(out, "▁", " ")
```

The high-level `Generate` path doesn't hit this because it doesn't
expose detokenize directly.

---

## `ScoreTexts` rejects `len(targets) > 1`

**Symptom.** `Session.ScoreTexts` returns
`INVALID_ARGUMENT: Target text size should be 1`.

**Cause.** The current LiteRT-LM CPU engine restricts
`run_text_scoring` to exactly one candidate per call. The Go API
accepts `[]string` for forward compatibility.

**Fix.** Score one target at a time. Open a fresh `Session` between
calls if needed (the engine doesn't allow re-prefill on a session
that has already scored or decoded).

---

## `Responses.Score(i)` returns `(0, true)` for non-scoring sources

**Symptom.** A `Responses` produced by `Session.GenerateContent` /
`Client.Generate` / `Session.RunDecode` reports `Score(0) == (0, true)`
— `ok=true` but the value is zero.

**Cause.** The C `has_score_at` predicate fires for any in-range
index, *not* only when a real score was computed. It's a "slot
exists" predicate. Non-scoring sources report `(0, true)` as a
placeholder.

**Fix.** Only treat `Score` values as meaningful when they came from
`Session.ScoreTexts`. The producing method is the source of truth
for whether a real score was computed.

---

## "Session reuse" engine error

**Symptom.**
`Failed to run decode: INTERNAL: new_step must be less than or equal
to TokenCount(), got X vs Y` — typically when running a second
prefill / decode / score on the same session.

**Cause.** The C engine restricts each `Session` to one prefill→decode
or prefill→score cycle. Trying to run a second cycle on the same
session corrupts internal state.

**Fix.** Open a fresh `Session` for each independent generation. The
high-level `Client.Generate` / `GenerateStream` / `GenerateResponse`
do this automatically — each call gets its own session.

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

The proper fix is upstream — rebuilding the prebuilt libs without
the profile flag. Until then, the env-var workaround is harmless.

---

## `engine_create` returns nil with `DYNAMIC_UPDATE_SLICE` errors

**Symptom.** `litertlm.New` (or low-level `NewEngine`) errors during
construction; the C-side log mentions `DYNAMIC_UPDATE_SLICE`.

**Cause.** `WithMaxTokens(n)` is set below the model's smallest
prefill signature (typically 128).

**Fix.** Use `WithMaxTokens(1024)` or higher. The high-level `Client`
defaults to 4096, which works for every Gemma 4 variant.

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

The conversation example's `toInt` helper covers this plus the
string fallback for cases where the model wraps numbers in quotes.

---

## Markers `<|"|>` in tool-call argument values

**Symptom.** A tool-call argument has the literal text
`<|"|>Boston, MA<|"|>` instead of `Boston, MA`.

**Cause.** Gemma 4's chat-template renderer leaves its internal quote
markers in string-typed arguments when the C side surfaces them as
JSON.

**Fix.** The high-level `*Reply` strips these automatically on
parse, so callers see clean values. If you're working at the low
level (`Conversation.SendMessage`), strip them yourself:

```go
strings.ReplaceAll(arg, `<|"|>`, "")
```
