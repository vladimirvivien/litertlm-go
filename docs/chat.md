# Chat

The `Chat` API provides support for multi-turn conversations with 
with optional system prompt and tool calling. 
The API wraps the C-side `Conversation` API, which applies the
model's chat template (e.g. Gemma's `<|turn>user … <turn|>`) 
appropriately for proper multi-turn conversations.

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
| `WithTool(defs ...)`            | One or more `ToolDefinition`s the model may call. Mix `RawTool` (hand-built) and `ManagedTool` (typed handler) freely. |
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

Two flavors of tool attach to a `Chat` via `WithTool`:

| Flavor        | Constructor    | Dispatch                                             |
|---------------|----------------|------------------------------------------------------|
| `RawTool`     | `NewRawTool`   | Manual — `Reply.ToolCalls()` + `Chat.SendToolResult` |
| `ManagedTool` | `RegisterTool` | Framework dispatches the typed handler               |

Both satisfy `ToolDefinition` and may be mixed in the same call:

```go
chat, _ := client.NewChat(ctx,
    litertlm.WithSystemPrompt("You are a calculator. Always call the tool."),
    litertlm.WithTool(calcAdd),
)
```

When the chat has at least one `ManagedTool` registered, `Chat.Send`
runs the dispatch loop and returns the post-tool natural-language
reply directly. Replies whose tool calls aren't all dispatchable
(unknown name or `RawTool`) come back for manual handling.

Cap the loop with `WithMaxToolHops(n)` (default 5). Override the
per-tool error-propagation policy with `WithToolPolicy(p)` at
`RegisterTool` time.

See the [Tools guide](tools.md) for the full reference: schema
reflection rules, dispatch semantics, mixed-registration behavior,
`ErrToolHopsExceeded` / `ToolHopsError`, and `ToolPolicy` modes.

### `ToolDefinition` and `ToolCall` types

```go
type ToolDefinition interface {
    Name() string
    Description() string
    Parameters() map[string]any  // JSON-Schema-shaped
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

- [Tools guide](tools.md) — full reference for `RawTool`,
  `ManagedTool`, `RegisterTool`, schema reflection, the dispatch
  loop, `WithMaxToolHops`, and `ToolPolicy`.
- [`examples/chat/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/chat)
  — minimal multi-turn demo.
- [`examples/autotool/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/autotool)
  — typed tool registration with auto-dispatch.
- [`examples/conversation/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/conversation)
  — manual dispatch with `NewRawTool` + `SendToolResult`.
- [Structured output](structured-output.md) — when you want
  type-safe JSON instead of free-form text.
- [Low-level API](low-level.md) — `Conversation`, `ConversationConfig`.
