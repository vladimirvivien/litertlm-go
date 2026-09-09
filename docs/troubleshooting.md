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

## Garbage or repetitive output from `Generate` / `GenerateStream`

**Symptom.** `Generate` (or `GenerateStream`, or `GenerateMulti`)
returns one of:

- An empty string, or `resp.Text(0) == ""`.
- A single token repeated until `-max` is hit
  (`0000000000…`, `\n\n\n…`, `<bos><bos>…`).
- A short hallucinated phrase repeated
  (`**{sea}** **{sea}** **{sea}**`).
- A coherent first sentence that then loops
  (`Marco who set out one morning to catch fish. Marco who set out
  one morning to catch fish. …`).

**Cause.** Chat-tuned models (Gemma instruct, Llama-Instruct, …) were
fine-tuned with the model's chat template wrapping every conversation
(e.g. Gemma 4:
`<bos><start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n`).
The raw `Generate` / `GenerateStream` / `GenerateMulti` paths send
your prompt to the model **as-is** — the chat template is *not*
applied. Without the framing tokens the model has no idea it's
supposed to answer a user, and degenerates: empty, a stuck token, or a
repetition trap. Larger chat-tuned models (Gemma 4 E4B) tend to fall
into loops; smaller ones (Gemma 4 E2B) sometimes manage to extend
completion-style prompts but still fail on bare instructions.

**Fix.** Use the high-level [`Chat`](chat.md) API. Both
`Chat.Send` and `Chat.SendStream` apply the model's chat template
before each turn:

```go
chat, _ := client.NewChat(ctx, litertlm.WithSystemPrompt("You are a friendly assistant."))
defer chat.Close()

reply, _ := chat.Send(ctx, "Explain why the sky is blue.")
fmt.Println(reply.Text())

// streaming form:
for chunk, err := range chat.SendStream(ctx, "Explain why the sky is blue.") {
    if err != nil { /* … */ }
    fmt.Print(chunk.Text)
}
```

The `Conversation` low-level API also applies the chat template.

If you genuinely need the raw `Generate` path (e.g. base model, or
pure text completion), pick a *completion-style* prompt the model can
extend rather than an instruction:

- `"The capital of France is"`
- `"Once upon a time in a small village by the sea, "`
- `"To install Go on Linux: 1)"`

Even then, expect chat-tuned E4B-class models to drift into loops on
long generations — the chat-templated path is the supported one for
chat-tuned models.

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

## `ErrUnimplemented`: `missing C symbol "..." in loaded library`

**Symptom.** A call into the Go API returns an error wrapping
`litertlm.ErrUnimplemented` (`*litertlm.CError` with `Code: StatusUnimplemented`):

```
litertlm: missing C symbol "litert_lm_session_config_set_apply_prompt_template"
in loaded library (refresh prebuilt LiteRT-LM libs)
```

`litertlm.Load` validates the major C API version (`1.x`) and returns
early; per-symbol lookup happens lazily on first use of a method, so
version-specific symbols fail with `ErrUnimplemented` when invoked
against a build that does not export them.

**Cause.** The LiteRT-LM library staged in `$LITERTLM_LIB` does not
export the named C API symbol — typically because it predates the
requested C API minor version.

**Fix.** Re-stage the prebuilt LiteRT-LM libraries from a current
upstream release (`v0.19.1` or newer) per
[`LITERTLM-BUILD.md`](https://github.com/vladimirvivien/litertlm-go/blob/main/LITERTLM-BUILD.md).

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
returns `litertlm: send: litertlm: conversation_send_message failed`
on the second turn of a Qwen3 conversation. `RenderMessage` on the
same turn returns `C side returned NULL`.

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

