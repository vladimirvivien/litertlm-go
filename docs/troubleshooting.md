# Troubleshooting

Known limitations and issues, with their root causes and resolutions.

## Library not found

**Symptom.** `litertlm.New` (or `litertlm.Load`) returns
`"litert-lm" not found; checked: ...` listing several paths.

**Cause.** The LiteRT-LM shared library is not at any of the locations
the loader searches.

**Search order (when `WithLib` is empty):**

1. `LITERTLM_LIB` environment variable.
2. Platform default paths:
    - Linux/macOS/FreeBSD: `$XDG_DATA_HOME/litertlm/lib`,
      `~/.litertlm/lib`, `/opt/litertlm/lib`,
      `/opt/homebrew/lib` (macOS only), `/usr/local/lib`.
    - Windows: `%LOCALAPPDATA%\litertlm\lib`,
      `%PROGRAMFILES%\litertlm\lib`.

The default library short-name is `litert-lm` (`liblitert-lm.so`,
`liblitert-lm.dylib`, `litert-lm.dll`), falling back to legacy
backend-specific names (`litertlm_c_cpu` / `litertlm_c`) when absent.
Override it with `WithLibName("...")` or `LITERTLM_LIB_NAME`.

**Fix.** Either:

- Pass `WithLib("/abs/path/to/lib")` to `litertlm.New`.
- Set `LITERTLM_LIB=/abs/path/to/lib`.
- Symlink your lib directory into one of the default paths above.

---

## Garbage or repetitive output from low-level `Session` APIs

**Symptom.** Low-level `Session.GenerateContent` or `Session.RunPrefill` / `Session.RunDecode` returns one of:

- An empty string, or `resp.Text(0) == ""`.
- A single token repeated until max output tokens is reached
  (`0000000000…`, `\n\n\n…`, `<bos><bos>…`).
- A short hallucinated phrase repeated
  (`**{sea}** **{sea}** **{sea}**`).
- A coherent first sentence that then loops
  (`Marco who set out one morning to catch fish. Marco who set out
  one morning to catch fish. …`).

**Cause.** Chat-tuned models (Gemma instruct, Qwen, Llama-Instruct, …) are
fine-tuned with a chat template wrapping every conversation turn
(for example, Gemma's `<|turn>user\n…<turn|>\n<|turn>model\n`).
Raw `Session.GenerateContent` and `Session.RunPrefill` / `Session.RunDecode`
forward your prompt bytes **as-is** unless `SessionConfig.SetApplyPromptTemplate(true)`
and a model template handle are active. Without template framing, chat-tuned
models frequently emit an immediate end-of-turn token or degenerate into repetition loops.

**Fix.** Use the high-level [`Client.Generate`](client.md#generatectx-prompt-opts), [`Client.GenerateStream`](client.md#generatestreamctx-prompt-opts), or [`Chat`](chat.md) APIs. Both `Client.Generate*` and `Chat.Send*` open a `Conversation` under the hood, applying the model's chat template automatically:

```go
chat, _ := client.NewChat(ctx, litertlm.WithSystemPrompt("You are a friendly assistant."))
defer chat.Close()

reply, _ := chat.Send(ctx, "Explain why the sky is blue.")
fmt.Println(reply.Text())
```

To curb repetition loops during open-ended generation, apply per-turn logit penalties with `WithRepetitionPenalty` and `WithNoRepeatNgram` (see [`examples/logit-penalties/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/logit-penalties)).

When driving raw `Session` prefill/decode directly without a template, supply *completion-style* prefixes the base distribution can extend (`"The capital of France is"`) or branch from checkpoints built with `SaveCheckpoint` / `RewindToCheckpoint`.

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
    $Env:LITERTLM_LIB = "$Env:USERPROFILE\litertlm\lib"
    go run main.go
    ```
---

## `engine_create` fails with `DYNAMIC_UPDATE_SLICE` errors

**Symptom.** `litertlm.New` (or low-level `NewEngine`) errors during
construction; the C-side log mentions `DYNAMIC_UPDATE_SLICE`.

**Cause.** `WithMaxTokens(n)` is set below the model's smallest
prefill signature (typically 128).

**Fix.** Use `WithMaxTokens(1024)` or higher. The high-level `Client`
defaults to 4096, which works for every Gemma 4 variant.

---

## `ErrUnimplemented`: `missing C symbol` or API minor-version mismatch

**Symptom.** A call into the Go API returns an error wrapping
`litertlm.ErrUnimplemented` (`*litertlm.CError` with `Code: StatusUnimplemented`):

```text
litertlm: missing C symbol "litert_lm_session_config_set_apply_prompt_template"
in loaded library (refresh prebuilt LiteRT-LM libs)
```

Or for features gated behind a newer C API minor version:

```text
litertlm: version_check: requires LiteRT-LM C API 1.1+, loaded library is 1.0.0 (Unimplemented)
```

`litertlm.Load` validates the major C API version (`1.x`) and returns
early; per-symbol lookup happens lazily on first use of a method, so
version-specific symbols fail with `ErrUnimplemented` when invoked
against a build that does not export them.

**Cause.** The LiteRT-LM library staged in `$LITERTLM_LIB` does not
export the requested symbol or reports an earlier C API minor version. Features requiring `1.1+` include `WithSelectedSignatures` / `EngineSettings.SetSelectedSignatures`, `WithToolChoice` / `OptionalArgs.SetToolChoice`, `PlanModelSync` / `SyncModel` / `SyncModelCache`, and `WithDebuggerEnabled` / `EngineSettings.SetEnableDebugger`. Speculative decoding `Lenience != 0` also requires a runtime exporting `litert_lm_speculative_decoding_config_set_lenience`.

**Fix.** Match error handling with `errors.Is(err, litertlm.ErrUnimplemented)`, leave optional fields at their defaults (`Lenience: 0`), or stage a newer LiteRT-LM shared library release (minimum supported base is `v0.19.1` per [`LITERTLM-BUILD.md`](https://github.com/vladimirvivien/litertlm-go/blob/main/LITERTLM-BUILD.md)).

---

## `Failed to create GemmaModelConstraintProvider`

**Symptom.** Opening a chat with `WithConstrainedDecoding(true)` (or `WithConstraintProvider(...)`) fails during `NewChat`:

```text
litertlm: NewChat: litertlm: conversation_create: INTERNAL: Failed to create GemmaModelConstraintProvider. (Internal)
```

**Cause.** Grammar-constrained decoding requires constraint-provider metadata embedded inside the `.litertlm` model file. Compact models without embedded constraint-provider metadata (for example, `gemma3-270m-it-q8.litertlm`) cannot initialize a constraint provider.

**Fix.** Use a model containing constraint metadata (such as `gemma3-1b-it-int4`, `gemma-4-E2B-it`, or `Qwen3-0.6B`) for grammar constraints, or use [`GenerateData[T]`](structured-output.md) without `WithConstrainedDecoding(true)`.

---

## `UNIMPLEMENTED: Sampler type: 3 not implemented yet`

**Symptom.** Creating a session or conversation with `SamplerGreedy` (`SamplerType = 3`, returned by `DefaultSamplerParams()`) fails on CPU with `StatusUnimplemented`.

**Cause.** Greedy sampler type `3` is not implemented on the CPU executor in LiteRT-LM `v0.19.1`.

**Fix.** Use `SamplerTopP` with `Temperature: 0` for deterministic sampling (`litertlm.SamplerParams{Type: litertlm.SamplerTopP, TopK: 40, TopP: 0.95, Temperature: 0, Seed: 1}`), or omit `SetSamplerParams` / `WithDefaultSampler` to use the model's built-in sampler configuration.

---

## GPU run falls back to CPU

**Symptom.** Running with `WithBackend("gpu")` logs
`WARNING: GPU accelerator could not be loaded and registered` and
inference continues on CPU.

**Cause.** On Windows, DirectX Shader Compiler binaries
(`dxcompiler.dll`, `dxil.dll`) are missing from `$LITERTLM_LIB`, or an
older split build is missing GPU accelerator plugins
(`libLiteRtWebGpuAccelerator.*`, `libLiteRtTopKWebGpuSampler.*`). Unified
`liblitert-lm` builds bundle the LiteRT GPU accelerator internally.

**Fix.** Stage `dxcompiler.dll` and `dxil.dll` on Windows (automatic
with `libfetch.Fetch(..., libfetch.WithBackend("gpu"))`), or re-stage the
shared library files per
[`LITERTLM-BUILD-WINDOWS.md`](https://github.com/vladimirvivien/litertlm-go/blob/main/LITERTLM-BUILD-WINDOWS.md)
and [`LITERTLM-BUILD.md`](https://github.com/vladimirvivien/litertlm-go/blob/main/LITERTLM-BUILD.md).

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

## `conversation_send_message failed` on Qwen3 multi-turn

**Symptom.** `Chat.Send` (or low-level `Conversation.SendMessage`)
returns an error wrapping `*CError` on `conversation_send_message`
on the second turn of a Qwen3 conversation, and `RenderMessage` on
the same turn fails in `conversation_render_message_to_string`.

**Cause.** Qwen3's Jinja chat template calls
`reasoning_content.strip('\n')` when rendering the prior assistant
turn's `<think>...</think>` block. LiteRT-LM's embedded Jinja
implementation does not provide a `string.strip(chars)` method, so
the template aborts before any decode runs. Reproducible against the
upstream `litert_lm_advanced_main.exe --multi_turns=true` binary, so
the bug is wholly in upstream LiteRT-LM.

**Fix.** None available in `litertlm-go`. Single-turn flows
(`Client.Generate`, `Client.GenerateStream`, `GenerateData[T]`) work
on Qwen3. Track upstream LiteRT-LM for a Jinja `strip(chars)`
implementation or a patched Qwen3 chat template.

Smaller Qwen3 variants (e.g. 0.6B) may render turn 2 without hitting
the failing template branch but can still emit `<|endoftext|>` and
`Human:` continuation patterns past their stop tokens on long
outputs.

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

---

## GPU process crashes on Windows (Exception 0xc0000005)

**Symptom.** Running GPU-backed inference crashes the test or application process with a memory access violation (`0xc0000005`).

**Cause.** Graphics API drivers and shader compilers (e.g., Direct3D 12/WebGPU loaded via `libwebgpu_dawn.dll`) cannot compile shaders or manage TLS state on the background worker threads managed by Go's runtime scheduler.

**Fix.** Execute client initialization and inference calls on the operating system's primary main thread. Avoid executing them inside concurrent goroutines or tests without thread locking (`runtime.LockOSThread()`).

---

## Failed to create engine (INTERNAL: ERROR: [unknown:0])

**Symptom.** Client initialization fails early during engine creation with:
`Failed to create engine: INTERNAL: ERROR: [unknown:0]`

**Cause.** Custom source builds splitting the C API from auxiliary GPU plugin libraries linked `libLiteRt` statically instead of dynamically, preventing delegate registration.

**Fix.** Prefer building upstream's unified target (`//c:litert-lm`) from [`LITERTLM-BUILD.md`](https://github.com/vladimirvivien/litertlm-go/blob/main/LITERTLM-BUILD.md). For split source builds with separate plugin libraries, compile with `--define=litert_runtime_link_mode=dynamic` and ensure `$LITERTLM_LIB` contains `libLiteRt` and `libwebgpu_dawn` on `PATH` / `LD_LIBRARY_PATH`.

