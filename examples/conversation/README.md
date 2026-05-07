# Example: Chat with Tools Calls

This examples shows the use of the **Chat API** with tool calls.

> See the basic [**Chat**](../chat/) example/

## What this example shows

- **Tool declaration** — `litertlm.NewRawTool` builds an OpenAI/Anthropic-style
  tool schema. The chat template renders it into the model's native
  tool-call markers.
- **Manual dispatch** — `Reply.ToolCalls()` surfaces the model's
  function-invocation request; the example dispatches it in Go.
- **Tool response message** — after running the function locally,
  `Chat.SendToolResult` posts the result back to the model so it
  can continue the turn.

For typed handlers and framework-managed dispatch (no manual
`SendToolResult` step), see `litertlm.RegisterTool`.

## Prerequisites

1. LiteRT-LM shared library files staged in`LITERTLM_LIB`.
2. A `.litertlm` model (i.e. Gemma 4). 
3. `litertlm-go`


## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/conversation \
    -model /abs/path/to/gemma-4-E4B-it.litertlm
    -prompt "What is 17 plus 25?"
```

| Flag       | Default                       | Notes                                                      |
| ---------- | ----------------------------- | ---------------------------------------------------------- |
| `-model`   | (required)                    | Path to a `.litertlm` chat-tuned model.                    |
| `-prompt`  | `"What is 17 plus 25?"`       | The user question. Should be one the model can solve via `calc_add`. |
| `-backend` | `"cpu"`                       | Set to `"gpu"` if you staged the GPU-capable build.        |
| `-lib`     | `$LITERTLM_LIB`               |                                                            |

### Expected output

```
user>     What is 17 plus 25?
(raw)     {"role":"assistant","tool_calls":[{"type":"function","function":{"name":"calc_add","arguments":{"a":17.0,"b":25.0}}}]}
(call)    calc_add(map[a:17 b:25])
(result)  map[result:42]
(raw)     {"role":"assistant","content":[{"type":"text","text":"The sum of 17 and 25 is 42."}]}
bot>      The sum of 17 and 25 is 42.
```
