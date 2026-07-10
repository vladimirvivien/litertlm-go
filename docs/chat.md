# Chat

The `Chat` API runs multi-turn conversations with optional system
prompt and tool calling. It wraps the C-side `Conversation` API,
which applies the model's chat template (e.g. Gemma's
`<|turn>user … <turn|>`) on every turn.

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

Open a `Chat` from a `Client` with `Client.NewChat(ctx, opts...)`.
Call `Close()` on the returned `*Chat` when done.

Chat configuration:

| Option                          | Effect                                                                                          |
|---------------------------------|-------------------------------------------------------------------------------------------------|
| `WithSystemPrompt(s)`           | The system instructions. Pass just the text — the library handles the message wrapping automatically. |
| `WithTool(defs ...)`            | Registers one or more tools the model can call. Mix hand-built (`RawTool`) and typed (`ManagedTool`) tools. |
| `WithInitialMessages(msgs)`     | Pre-seeds the conversation history with prior turns (supports both text and multimodal parts). |
| `WithConstrainedDecoding(on)`   | Toggles constrained-decoding mode. |
| `WithExtraContext(json)`        | Adds optional extra context (in JSON format) to the conversation preface. |
| `WithFilterChannelContentFromKVCache(on)` | Excludes model reasoning-channel tokens from the KV cache to conserve cache space. |

## `Send(ctx, message)` and `Reply`

`Send` issues a synchronous user-role message and returns a `*Reply`.

```go
type Reply struct{ /* unexported */ }
func (r *Reply) Text() string         // concatenated text content parts
func (r *Reply) ToolCalls() []ToolCall // structured function-call requests
func (r *Reply) HasToolCalls() bool
func (r *Reply) Raw() string          // original C-side JSON, for debugging
```

## `SendStream(ctx, message)`

`SendStream` is the streaming variant of `Send`. It returns an
`iter.Seq2[Chunk, error]` over the model's response chunks as they
arrive.

```go
for chunk, err := range chat.SendStream(ctx, message) {
    if err != nil { break }
    fmt.Print(chunk.Text)
}
```

## Multimodal turns: `SendMulti` and `SendMultiStream`

Send image and audio inputs through the same `Chat` handle as text
turns. `SendMulti` accepts `[]Part` instead of `string`; the
underlying `Conversation` accumulates KV state across turns
regardless of modality, so follow-up text turns can reference
earlier multimodal content.

```go
img, err := litertlm.ImageFromFile("photo.jpg")
// ...
reply, err := chat.SendMulti(ctx, []litertlm.Part{
    img,
    litertlm.Text("Describe this image in one sentence."),
})

// Follow-up text turn — same Chat, image embeddings still in KV cache:
reply, err = chat.Send(ctx, "What's the dominant color?")
```

`SendMultiStream` is the streaming sibling:

```go
for chunk, err := range chat.SendMultiStream(ctx, parts) {
    if err != nil { break }
    fmt.Print(chunk.Text)
}
```

Requirements:

- Image Parts require `WithVisionBackend` on the Client at `New`
  time. Audio Parts require `WithAudioBackend`. Calling `SendMulti`
  with the wrong backend (or no backend) surfaces a C-side
  `conversation_create` failure.
- An empty `[]Part` is rejected up front with `litertlm: SendMulti:
  empty parts`. Pass at least one Part.
- Tool dispatch behaves identically to `Send` — the model can emit
  `tool_call` content in response to multimodal input, and
  `SendMulti` runs the auto-dispatch loop the same way.
- A `[]Part` containing only text Parts is equivalent to `Send`
  with the text concatenated; the multimodal path is only needed
  when at least one image / audio Part is present.

Contrast with `Client.GenerateMulti` / `GenerateMultiStream` /
`GenerateMultiResponse`: those are one-shot calls that open a
fresh `Conversation`, run one inference, and discard it. KV state
does not persist between calls. Use `Chat.SendMulti*` when you
want successive turns to share the same conversation state.

## Per-call options

All five `Chat.Send*` methods accept variadic `RuntimeOption` values
applied to a single turn (and to every dispatch hop the turn
triggers).

`WithVisualTokenBudget(n)` caps the vision tokens consumed by a
`SendMulti` / `SendMultiStream` turn. Text-only turns ignore it.
Effective on Gemma 4 vision-enabled models.

```go
reply, err := chat.SendMulti(ctx, []litertlm.Part{img, litertlm.Text("...")},
    litertlm.WithVisualTokenBudget(512),
)
```

`WithReturnToolRequests(true)` bypasses the auto-dispatch loop on
`Send`, `SendMulti`, and `SendToolResult`. The first reply
containing tool calls is returned directly via `Reply.ToolCalls()`
even when every call maps to a registered `ManagedTool`. Pair with
`Chat.SendToolResult` to feed the result back. Streaming methods
ignore this flag — `SendStream` / `SendMultiStream` always run the
dispatch loop.

```go
reply, err := chat.Send(ctx, "What's the weather in Paris?",
    litertlm.WithReturnToolRequests(true),
)
for _, call := range reply.ToolCalls() {
    // inspect call.Function.Name / Arguments before dispatching manually
}
```

`WithMaxConcurrentTools(n)` runs tool handlers in parallel within a
single dispatch hop. `n <= 1` keeps the default sequential dispatch;
`n > 1` caps in-flight handlers at `n`. Result ordering in the
follow-up tool-role message matches the model's original call order
regardless of completion order. The first handler to fail (in real
time) terminates the batch; sibling handlers see ctx cancellation
and may bail.

```go
reply, err := chat.Send(ctx, "Compare weather in Paris and Tokyo",
    litertlm.WithMaxConcurrentTools(4),
)
```

Tool handlers must be safe to invoke from multiple goroutines when
`n > 1`. Shared mutable state in a closure-captured `ManagedTool`
handler needs its own synchronization.

`WithMaxOutputTokens(n)` caps output tokens produced for this turn.

```go
reply, err := chat.Send(ctx, "Write a long story",
    litertlm.WithMaxOutputTokens(50),
)
```

`WithSampler` is not per-call on Chat — it applies at `NewChat` time on the Client's session config.

## Template Preface Rendering

To inspect the system prompts, initial message wrapper formatting, or tools schema structure exactly as they will be formatted for the model (before any user input is appended), call `RenderPreface()` on the underlying `Conversation`:

```go
rendered, err := chat.Conversation().RenderPreface()
if err != nil {
    return err
}
fmt.Println(rendered)
// Output: <|im_start|>system\nYou are a helpful assistant...
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

Seeding history with `WithInitialMessages` accepts both text-only and
multimodal turns:

```go
img, _ := litertlm.ImageFromFile("photo.jpg")
chat, _ := client.NewChat(ctx,
    litertlm.WithInitialMessages([]litertlm.Message{
        {Role: "user", Parts: []litertlm.Part{img, litertlm.Text("What is this?")}},
        {Role: "assistant", Parts: []litertlm.Part{litertlm.Text("A wooden table with a lamp.")}},
    }),
)
chat.Send(ctx, "What's the dominant color?")  // resumes with image in KV
```

## Introspection

`Chat.TokenCount()` returns the number of tokens currently held in the
underlying conversation's KV cache (prefill + decode), accumulated
across every turn — including tool-dispatch hops. Use it to project a
chat against the engine's max-token budget.

```go
n, err := chat.TokenCount() // tokens in the KV cache
```

It does not require `WithBenchmarkEnabled`. For a per-turn prefill /
decode breakdown, read `Conversation.BenchmarkInfo()` instead (that
path does require benchmark collection).

Requires LiteRT-LM v0.13.1 or newer.

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
