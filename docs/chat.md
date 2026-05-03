# Chat

The `Chat` API provides support for multi-turn conversations with 
with optional system prompt and tool calling. 
The API wraps the C-side `Conversation` API, which applies the
model's chat template (e.g. Gemma's `<|turn>user … <turn|>`) for 
proper multi-turn conversations.

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

## Creating `NewChat(ctx, opts...)`

You start your chat sessions from a `Client` instance. Once you have a chat instance, always remember to `Close()` it when done.

Chat configuration:

| Option                          | Effect                                                                                          |
|---------------------------------|-------------------------------------------------------------------------------------------------|
| `WithSystemPrompt(s)`           | The system message. **Pass just the content** — the C side wraps it in a `{role,content}` envelope itself. |
| `WithTools(tools)`              | A slice of `litertlm.Tool` declarations the model may call.                                     |
| `WithInitialMessages(msgs)`     | Seed history with prior turns.                                                                  |
| `WithConstrainedDecoding(on)`   | Toggle the engine's constrained-decoding mode (boolean only — schema delivery is upstream-pending). |

## `Send(ctx, message)` and `Reply`

Use the `Send` method to synchronously send user-role messages. Each send returns a `*Reply`
which gives you access to chat session resources.

```go
type Reply struct{ /* unexported */ }
func (r *Reply) Text() string         // concatenated text content parts
func (r *Reply) ToolCalls() []ToolCall // structured function-call requests
func (r *Reply) HasToolCalls() bool
func (r *Reply) Raw() string          // original C-side JSON, for debugging
```

## `SendStream(ctx, message)`

`SendStream` is the Streaming variant of `Send`. It returns an iterator of type `iter.Seq2[Chunk, error]`.
This allows programs to easily accessing streamed replies from the engine.

```go
for chunk, err := range chat.SendStream(ctx, message) {
    if err != nil { break }
    fmt.Print(chunk.Text)
}
```

## Tool calling
Tool calling starts with the definition of a `litertlm.Tool` which is then
handed the `Client` when creating a new chat.

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

    // Execute requested tool call.
    a := call.Function.Arguments["a"].(float64)
    b := call.Function.Arguments["b"].(float64)
    result := map[string]int{"result": int(a + b)}

    // Send the tool-call result back as a tool-role message.
    final, _ := chat.SendToolResult(ctx, call.Function.Name, result)
    fmt.Println(final.Text())  // "The sum of 17 and 25 is 42."
}
```

### `Tool` and `ToolCall` types
The Parameters `map` is similar to OpenAI / Anthropic function-calling
schemas;

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

## Multi-turn

A single `Chat` instance preserves history across calls. Successive
`Send` calls let the model see prior turns:

```go
chat.Send(ctx, "I'm planning a trip to Tokyo.")
chat.Send(ctx, "Suggest a 3-day itinerary.")  // model knows "Tokyo"
chat.Send(ctx, "What about the third day?")   // model knows the prior itinerary
```

When you need a fresh context, open a new `Chat`.

## See also

- [`examples/chat/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/chat)
  — minimal multi-turn demo.
- [`examples/conversation/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/conversation)
  — full tool-calling walkthrough.
- [Structured output](structured-output.md) — when you want
  type-safe JSON instead of free-form text.
- [Low-level API](low-level.md) — `Conversation`, `ConversationConfig`.
