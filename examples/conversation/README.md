# conversation — system + tools + structured tool_calls

A self-contained tool-using agent built on the **Conversation API**. The
"tool" is just Go-side integer addition, so the example runs offline and
without any external dependencies — all you need is a chat-tuned
`.litertlm` model that supports function calling (Gemma 4 instruct works
out of the box).

## What this example shows

Beyond the basic [`chat`](../chat/) demo:

- **Tool declaration** via the `tools_json` slot of
  `NewConversationConfig`. The C side renders the OpenAI/Anthropic-style
  schema into the model's native tool-call format using the chat
  template baked into the `.litertlm` file.
- **Structured `tool_calls` in the assistant reply.** When the model
  decides to call a function it emits Gemma 4's `<|tool_call>...<tool_call|>`
  markers; the C side parses them and surfaces a JSON `tool_calls` array
  on the assistant message — no regex needed on the Go side.
- **Tool-role response message.** After running the function locally,
  the example sends the result back via
  ```json
  {"role":"tool","content":[{"name":"calc_add","response":{"result":42}}]}
  ```
  which the chat template renders as
  `<|tool_response>response:calc_add{result:42}<tool_response|>` inside
  the same model turn the tool call opened.
- **System prompt via `systemMessageJSON`.** Pass just the content
  (string or content array), not a `{role,content}` envelope — the C
  side wraps it itself.

## Prerequisites

1. Native shared library + `libGemmaModelConstraintProvider.*` staged in
   a directory pointed to by `LITERTLM_LIB`.
2. A chat-tuned `.litertlm` model file with function-calling support.
   Gemma 4 instruct works:
   - `litert-community/gemma-4-E2B-it-litert-lm`
   - `litert-community/gemma-4-E4B-it-litert-lm`
3. Go 1.22+.

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/conversation \
    -model /abs/path/to/gemma-4-E4B-it.litertlm
```

Override the question with `-prompt`:

```bash
go run ./examples/conversation -model … -prompt "What is 100 plus 23?"
```

| Flag       | Default                       | Notes                                                      |
| ---------- | ----------------------------- | ---------------------------------------------------------- |
| `-model`   | (required)                    | Path to a `.litertlm` chat-tuned model.                    |
| `-prompt`  | `"What is 17 plus 25?"`       | The user question. Should be one the model can solve via `calc_add`. |
| `-backend` | `"cpu"`                       | Set to `"gpu"` if you staged the GPU-capable build.        |
| `-lib`     | `$LITERTLM_LIB`               |                                                            |

## Expected output

```
user>     What is 17 plus 25?
(raw)     {"role":"assistant","tool_calls":[{"type":"function","function":{"name":"calc_add","arguments":{"a":17.0,"b":25.0}}}]}
(call)    calc_add(map[a:17 b:25])
(result)  map[result:42]
(raw)     {"role":"assistant","content":[{"type":"text","text":"The sum of 17 and 25 is 42."}]}
bot>      The sum of 17 and 25 is 42.
```

The two `(raw)` lines are the JSON returned by `SendMessage` so you can
see exactly what the C side surfaces. The `(call)` line is the parsed
tool invocation, and `(result)` is what the local Go function returned.

## Notes

- **`tool_calls.arguments` types.** `encoding/json` decodes JSON numbers
  into `float64` by default, which is why the raw JSON shows `17.0`
  rather than `17`. The example's `toInt` helper covers that path. For
  string-typed arguments, the C side may also leave the model's native
  Gemma 4 quote markers (`<|"|>`) inside the value — `toInt` strips them
  before parsing, and a real-world tool executor should do the same for
  string fields it needs clean (see
  `litertlm-intro/weather-tool-convo` for one approach).
- **Single-turn vs multi-turn.** The Conversation handle keeps history
  internally, so a follow-up `SendMessage("Now add 5 to that")` would
  trigger a second tool call without re-sending the original question.
  Add a turn loop to extend the example.
- **Streaming.** Replace `conv.SendMessage(...)` with
  `conv.SendMessageStreamCh(messageJSON, "")` to receive incremental
  chunks; see [`stream`](../stream/) for the channel idiom.
- **Why `set_*` setters and not the all-args constructor?** The C API
  declares `litert_lm_conversation_config_create()` with no arguments
  (see `c/engine.h`); fields are populated via per-field setters. The
  Go binding (`pkg/litertlm/conversation.go:NewConversationConfig`)
  handles that internally — callers just pass the JSON strings.
