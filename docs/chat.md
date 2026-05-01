# Chat

Multi-turn conversations with optional system prompt and tool
calling. Built on the C-side Conversation API, which applies the
model's chat template (e.g. Gemma's `<|turn>user … <turn|>`) so the
bot output looks like a proper assistant reply.

```go
chat, err := client.NewChat(ctx,
    litertlm.WithSystemPrompt("You are a friendly assistant."),
)
defer chat.Close()

reply, err := chat.Send(ctx, "Hi, what is your name?")
fmt.Println(reply.Text())

reply, err = chat.Send(ctx, "Tell me a fun fact.")
fmt.Println(reply.Text())
```

`Chat` keeps dialogue history internally — successive `Send` calls
have access to prior turns.

## `NewChat(ctx, opts...)`

Construct from a `Client`. Always `Close()` when done.

| Option                          | Effect                                                                                          |
|---------------------------------|-------------------------------------------------------------------------------------------------|
| `WithSystemPrompt(s)`           | The system message. **Pass just the content** — the C side wraps it in a `{role,content}` envelope itself. |
| `WithTools(tools)`              | A slice of `litertlm.Tool` declarations the model may call.                                     |
| `WithInitialMessages(msgs)`     | Seed history with prior turns.                                                                  |
| `WithConstrainedDecoding(on)`   | Toggle the engine's constrained-decoding mode (boolean only — schema delivery is upstream-pending). |

!!! warning "System prompt shape"

    `WithSystemPrompt(s)` takes the **content** string. Don't pass a
    full `{"role":"system","content":"..."}` envelope; the C-side
    wrapping makes the chat template silently drop the prompt
    (verified failure mode). Plain text is what you want:

    ```go
    litertlm.WithSystemPrompt("You are a calculator assistant.")
    ```

## `Send(ctx, message)`

Synchronous user-role message. Returns a `*Reply`.

```go
type Reply struct{ /* unexported */ }
func (r *Reply) Text() string         // concatenated text content parts
func (r *Reply) ToolCalls() []ToolCall // structured function-call requests
func (r *Reply) HasToolCalls() bool
func (r *Reply) Raw() string          // original C-side JSON, for debugging
```

`ctx` cancellation flows through to `Conversation.Cancel` internally.

## `SendStream(ctx, message)`

Streaming variant. Returns `iter.Seq2[Chunk, error]` — same shape as
`Client.GenerateStream`. Tool-using replies arrive as raw text chunks
with the model's native tool-call markers; for structured `tool_calls`,
prefer `Send`.

```go
for chunk, err := range chat.SendStream(ctx, message) {
    if err != nil { break }
    fmt.Print(chunk.Text)
}
```

## Tool calling

```go
tools := []litertlm.Tool{
    {
        Type: "function",
        Function: litertlm.ToolFunction{
            Name:        "calc_add",
            Description: "Add two integers and return their sum.",
            Parameters: map[string]any{
                "type": "object",
                "properties": map[string]any{
                    "a": map[string]any{"type": "integer"},
                    "b": map[string]any{"type": "integer"},
                },
                "required": []string{"a", "b"},
            },
        },
    },
}

chat, _ := client.NewChat(ctx,
    litertlm.WithSystemPrompt("You are a calculator. Always call the tool."),
    litertlm.WithTools(tools),
)
defer chat.Close()

reply, _ := chat.Send(ctx, "What is 17 plus 25?")
if reply.HasToolCalls() {
    call := reply.ToolCalls()[0]
    // call.Function.Name        == "calc_add"
    // call.Function.Arguments   == map[string]any{"a": 17.0, "b": 25.0}

    // Execute locally.
    a := call.Function.Arguments["a"].(float64)
    b := call.Function.Arguments["b"].(float64)
    result := map[string]int{"result": int(a + b)}

    // Send the result back as a tool-role message.
    final, _ := chat.SendToolResult(ctx, call.Function.Name, result)
    fmt.Println(final.Text())  // "The sum of 17 and 25 is 42."
}
```

### `Tool` and `ToolCall` types

```go
type Tool struct {
    Type     string       // always "function"
    Function ToolFunction
}

type ToolFunction struct {
    Name        string
    Description string
    Parameters  map[string]any  // JSON-Schema-shaped
}

type ToolCall struct {
    Type     string
    Function ToolCallFunction
}

type ToolCallFunction struct {
    Name      string
    Arguments map[string]any  // numbers come as float64 (encoding/json default)
}
```

The Parameters map mirrors OpenAI / Anthropic function-calling
schemas; the C-side chat template renders them into the model's
native declaration format.

### `SendToolResult(ctx, name, result)`

`result` is JSON-marshaled directly — pass a struct or `map[string]any`
so the C-side template renders the response object faithfully. The
chat template expects the response field to be a JSON object, not a
string.

## Argument-quote stripping

Some tool-using models (especially smaller ones on Gemma 4) leak
their internal quote markers (`<|"|>`) into string-typed argument
values:

```
{"location": "<|\"|>Boston, MA<|\"|>"}
```

`*Reply` strips these markers automatically on parse, so by the time
your code reads `Arguments["location"]` you get clean
`"Boston, MA"`. Numeric and boolean arguments are passed through
unchanged.

## Multi-turn

A single `Chat` handle preserves history across calls. Successive
`Send` calls let the model see prior turns:

```go
chat.Send(ctx, "I'm planning a trip to Tokyo.")
chat.Send(ctx, "Suggest a 3-day itinerary.")  // model knows "Tokyo"
chat.Send(ctx, "What about the third day?")   // model knows the prior itinerary
```

If you need a fresh context, open a new `Chat`.

## See also

- [`examples/chat/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/chat)
  — minimal multi-turn demo.
- [`examples/conversation/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/conversation)
  — full tool-calling walkthrough.
- [Structured output](structured-output.md) — when you want
  type-safe JSON instead of free-form text.
- [Low-level API](low-level.md) — `Conversation`, `ConversationConfig`.
