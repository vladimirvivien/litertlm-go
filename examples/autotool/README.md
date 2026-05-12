# Example: Typed Tools with Auto-Dispatch

Demonstrates `litertlm.RegisterTool` + `WithTool`: define a tool with
a typed Go handler, attach it to a chat, and let the framework run
the tool-call → invoke → tool-result → next-turn loop. A single
`Chat.Send` call returns the post-tool natural-language answer.

## What this example shows

- **Typed tool definition.** `RegisterTool` reflects over the input
  struct (`AddIn`) to build a JSON-Schema parameters map; the
  description tag becomes the per-field description in the schema.
- **Auto-dispatch.** When the model invokes the tool, the framework
  unmarshals the arguments into a fresh `AddIn`, runs the handler,
  marshals the typed `AddOut`, and sends the result back. The loop
  continues until the model produces a text-only reply.
- **`*ToolHopsError`** with `errors.As` for inspecting partial state
  if the dispatch loop hits the cap.

For the lower-level path — hand-built tool declarations and manual
result handling with `Reply.ToolCalls()` + `Chat.SendToolResult` —
see [`examples/conversation/`](../conversation).

## Prerequisites

1. LiteRT-LM shared library files staged in `LITERTLM_LIB`.
2. A `.litertlm` chat-tuned model (e.g. Gemma 4).

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/autotool \
    -model /abs/path/to/gemma-4-E4B-it.litertlm \
    -prompt "What is 17 plus 25?"
```

| Flag       | Default                     | Notes                                     |
| ---------- | --------------------------- | ----------------------------------------- |
| `-model`   | (required)                  | Path to a `.litertlm` chat-tuned model.   |
| `-prompt`  | `"What is 17 plus 25?"`     | The user question.                        |
| `-backend` | `"cpu"`                     | Set to `"gpu"` for the GPU-capable build. |
| `-lib`     | `$LITERTLM_LIB`             |                                           |

## Expected output

```
user>     What is 17 plus 25?
(invoke)  calc_add(a=17, b=25)
bot>      The sum of 17 and 25 is 42.
```

The `(invoke)` line shows the handler firing inside the dispatch
loop; the caller issues one `Send` and reads one `Text()` for the
final natural-language reply.

## See also

- [Tools guide](../../docs/tools.md) — the full reference for typed
  and raw tools, dispatch semantics, hop cap, and `ToolPolicy`.
- [`examples/conversation/`](../conversation) — the manual-dispatch
  path with `NewRawTool` + `Chat.SendToolResult`.
