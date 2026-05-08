# Example: Manual Tool Dispatch

Demonstrates the lower-level tool-calling path: declare a tool with
`NewRawTool`, observe the model's tool call via `Reply.ToolCalls()`,
execute it in Go, and post the result back with
`Chat.SendToolResult`.

For typed handlers and framework-managed dispatch — a single
`Chat.Send` call with no `SendToolResult` step — see
[`examples/autotool/`](../autotool).

## What this example shows

- **Hand-built tool declaration** with `litertlm.NewRawTool` — name,
  description, and an OpenAI/Anthropic-style JSON-Schema
  `parameters` map.
- **Manual dispatch** — `Reply.ToolCalls()` surfaces the model's
  function-invocation request; the example's `executeToolCall`
  function runs it.
- **Result handoff** — `Chat.SendToolResult` posts the result back to
  the model so it can produce the final natural-language answer.

This pattern is the right fit when:

- The schema is generated dynamically (not derivable from a Go struct).
- The handler lives outside the chat lifetime (a long-running
  service, a remote endpoint).
- You want to inspect or transform the call before dispatching it.

## Prerequisites

1. LiteRT-LM shared library files staged in `LITERTLM_LIB`.
2. A `.litertlm` chat-tuned model (e.g. Gemma 4).

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/conversation \
    -model /abs/path/to/gemma-4-E4B-it.litertlm \
    -prompt "What is 17 plus 25?"
```

| Flag       | Default                  | Notes                                                              |
| ---------- | ------------------------ | ------------------------------------------------------------------ |
| `-model`   | (required)               | Path to a `.litertlm` chat-tuned model.                            |
| `-prompt`  | `"What is 17 plus 25?"`  | The user question. Should be one the model can solve via `calc_add`. |
| `-backend` | `"cpu"`                  | Set to `"gpu"` if you staged the GPU-capable build.                |
| `-lib`     | `$LITERTLM_LIB`          |                                                                    |

## Expected output

```
user>     What is 17 plus 25?
(call)    calc_add(map[a:17 b:25])
(result)  map[result:42]
bot>      The sum of 17 and 25 is 42.
```

## See also

- [Tools guide](../../docs/tools.md) — full reference for both tool
  flavors.
- [`examples/autotool/`](../autotool) — typed handler + auto-dispatch.
