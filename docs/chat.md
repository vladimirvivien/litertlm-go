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

Two flavors of tool can be attached to a Chat. Both implement
`ToolDefinition` and may be mixed in the same `WithTool` call:

| Flavor       | Constructor      | Dispatch                                             |
|--------------|------------------|------------------------------------------------------|
| `RawTool`    | `NewRawTool`     | Manual — `Reply.ToolCalls()` + `Chat.SendToolResult` |
| `ManagedTool`| `RegisterTool`   | Framework dispatches the typed handler               |

### `RawTool` — hand-built declaration

Pass a hand-rolled JSON-Schema `parameters` map:

```go
calcAdd := litertlm.NewRawTool(
    "calc_add",
    "Add two integers and return their sum.",
    map[string]any{
        "type": "object",
        "properties": map[string]any{
            "a": map[string]any{"type": "integer"},
            "b": map[string]any{"type": "integer"},
        },
        "required": []string{"a", "b"},
    },
)

chat, _ := client.NewChat(ctx,
    litertlm.WithSystemPrompt("You are a calculator. Always call the tool."),
    litertlm.WithTool(calcAdd),
)
defer chat.Close()

reply, _ := chat.Send(ctx, "What is 17 plus 25?")
if reply.HasToolCalls() {
    call := reply.ToolCalls()[0]
    // call.Function.Name        == "calc_add"
    // call.Function.Arguments   == map[string]any{"a": 17.0, "b": 25.0}

    a := call.Function.Arguments["a"].(float64)
    b := call.Function.Arguments["b"].(float64)
    result := map[string]int{"result": int(a + b)}

    final, _ := chat.SendToolResult(ctx, call.Function.Name, result)
    fmt.Println(final.Text())  // "The sum of 17 and 25 is 42."
}
```

### `ManagedTool` — typed handler

`RegisterTool` reflects over the input struct to build the
`parameters` schema, registers the tool on the Client, and returns a
`*ManagedTool[I, O]` that the framework dispatches automatically.

```go
type WeatherInput struct {
    Location string `description:"city and state, e.g. 'Boston, MA'"`
}
type WeatherOutput struct {
    Forecast string `json:"forecast"`
}

weather, _ := litertlm.RegisterTool(client, "get_weather",
    "Fetch a weather forecast",
    func(ctx context.Context, in WeatherInput) (WeatherOutput, error) {
        return WeatherOutput{Forecast: forecastFor(in.Location)}, nil
    })

chat, _ := client.NewChat(ctx, litertlm.WithTool(weather, calcAdd))
```

Field rules for the input struct:

- Exported fields only.
- Field name from `json:"name"` tag, else lowercased Go name. `json:"-"` excludes.
- Description from `description:"..."` tag, optional.
- Pointer fields are optional; non-pointer fields are required.
- Supported kinds: string, bool, all int/uint/float widths, slice/array, nested struct, pointer (unwrapped). Nesting capped at depth 32.

### Auto-dispatch loop

When the chat has at least one `ManagedTool` registered, `Chat.Send`
runs the tool-call → invoke → tool-result → next-turn loop until the
model produces a text-only reply, then returns that reply. Single
call from the user, single reply with the post-tool-call answer.

Behavior in mixed registrations:

- If every tool call in a model reply is `ManagedTool` → dispatch all
  sequentially; bundle results into a single tool-role message; loop.
- If any call is unknown or maps to a `RawTool` → return the reply
  for manual handling. The whole turn is yours.

Cap the loop with `WithMaxToolHops(n)` (default 5). Exceeding the cap
returns an error matching `errors.Is(err, ErrToolHopsExceeded)`. Use
`errors.As` to a `*ToolHopsError` for the partial last reply.

```go
chat, _ := client.NewChat(ctx,
    litertlm.WithTool(weather),
    litertlm.WithMaxToolHops(3),
)

reply, err := chat.Send(ctx, "what's the weather in Boston?")
if errors.Is(err, litertlm.ErrToolHopsExceeded) {
    var hops *litertlm.ToolHopsError
    errors.As(err, &hops)
    log.Printf("model still calling tools after %d hops; partial reply: %v",
        hops.Hops, hops.LastReply.Raw())
}
```

`SendStream` keeps the raw streaming behavior — it does not wrap the
dispatch loop. For typed-tool flows that need streaming, use `Send`
and stream the post-dispatch reply yourself.

### `ToolPolicy` — error handling per tool

Each `ManagedTool` carries a `ToolPolicy` set with `WithToolPolicy`
at registration time. The policy controls what happens when the
handler returns an error during dispatch:

| Policy                       | Behavior                                                                    |
|------------------------------|-----------------------------------------------------------------------------|
| `ToolPolicyReturnOnError` (default) | Propagate the error from `Chat.Send`; loop ends. Model is not informed. |
| `ToolPolicyInformOnError`    | Marshal `{"error": "<err.Error()>"}` as the tool's response and continue. The model can retry, apologize, or pick a different tool. |

```go
// Default: a validation error stops the loop and surfaces to the caller.
validate, _ := litertlm.RegisterTool(client, "validate", "...", validateHandler)

// Inform-on-error: transient failures the model can react to.
weather, _ := litertlm.RegisterTool(client, "get_weather", "...", weatherHandler,
    litertlm.WithToolPolicy(litertlm.ToolPolicyInformOnError))
```

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

- [`examples/chat/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/chat)
  — minimal multi-turn demo.
- [`examples/conversation/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/conversation)
  — full tool-calling walkthrough.
- [Structured output](structured-output.md) — when you want
  type-safe JSON instead of free-form text.
- [Low-level API](low-level.md) — `Conversation`, `ConversationConfig`.
