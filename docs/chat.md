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

| Option                                  | Effect                                                                                          |
|-----------------------------------------|-------------------------------------------------------------------------------------------------|
| `WithSystemPrompt(s)`                   | The system instructions. Pass just the text — the library handles the message envelope automatically. |
| `WithTool(defs ...)`                    | Registers one or more tools the model can call. Mix hand-built (`RawTool`) and typed (`ManagedTool`) tools. |
| `WithInitialMessages(msgs)`             | Pre-seeds the conversation history with prior turns (supports both text and multimodal parts). |
| `WithConstrainedDecoding(on)`           | Enables constrained-decoding mode for schema/regex grammars or `ToolChoice`. |
| `WithConstraintProvider(provider)`      | Selects the constraint grammar provider (`ConstraintProviderLLGuidance` or `ConstraintProviderNone`). |
| `WithPromptTemplate(tmpl)`              | Overrides the model's default chat prompt template. |
| `WithThinking(cfg)`                     | Sets conversation-wide thinking mode and token budget (`ThinkingConfig{Enabled, TokenBudget}`). |
| `WithExtraContext(json)`                | Adds optional extra context (in JSON format) to the conversation preface. |
| `WithFilterChannelContentFromKVCache(on)` | Excludes model reasoning-channel tokens from the KV cache to conserve cache space. |
| `WithMaxToolHops(n)`                    | Caps the auto-dispatch tool execution loop iterations (defaults to 5). |

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

`WithTurnThinking(cfg)` toggles reasoning or adjusts the thinking token budget on a single turn. Pass `TokenBudget: 0` to keep the C-side unlimited (`-1`) default, or pass a positive budget:

```go
reply, err := chat.Send(ctx, "Compare merge sort and quicksort.",
    litertlm.WithTurnThinking(litertlm.ThinkingConfig{Enabled: true, TokenBudget: 256}),
)
```

See [`examples/thinking/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/thinking).

`WithConstraint(typ, pattern)` enforces a `ConstraintJSONSchema` or `ConstraintRegex` grammar for a single turn. Open the chat with `WithConstrainedDecoding(true)` (and optionally `WithConstraintProvider(ConstraintProviderLLGuidance)`):

```go
chat, err := client.NewChat(ctx,
    litertlm.WithConstrainedDecoding(true),
    litertlm.WithConstraintProvider(litertlm.ConstraintProviderLLGuidance),
)
reply, err := chat.Send(ctx, "Return Ada Lovelace's birth year as JSON.",
    litertlm.WithConstraint(litertlm.ConstraintJSONSchema, `{"type":"object","properties":{"year":{"type":"integer"}},"required":["year"]}`),
)
```

Constrained decoding requires a `.litertlm` model with embedded constraint-provider metadata. See [`examples/constrained-json/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/constrained-json).

`WithToolChoice(choice)` enforces per-turn tool selection under constrained decoding (`ToolChoiceAuto()`, `ToolChoiceNone()`, `ToolChoiceRequired()`, `ToolChoiceFunction(name)`, `ToolChoiceAllowedTools(mode, names...)`, `ToolChoiceRawJSON(raw)`). Requires LiteRT-LM C API `1.1+`; returns `ErrUnimplemented` on C API `1.0.0`. See [Tools → Controlling Tool Selection](tools.md#controlling-tool-selection-withtoolchoice) and [`examples/tool-choice/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/tool-choice).

`WithRepetitionPenalty(cfg)`, `WithNoRepeatNgram(cfg)`, and `WithSuppressTokens(tokenIDs...)` adjust output logits per turn to curb loops or forbid specific token IDs:

```go
reply, err := chat.Send(ctx, prompt,
    litertlm.WithRepetitionPenalty(litertlm.RepetitionPenaltyConfig{
        RepetitionPenalty: 1.15,
        PresencePenalty:   0.2,
        WindowSize:        64,
    }),
    litertlm.WithNoRepeatNgram(litertlm.NoRepeatNgramConfig{NgramSize: 4, WindowSize: 64}),
    litertlm.WithSuppressTokens(badTokenID),
)
```

See [`examples/logit-penalties/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/logit-penalties).

`WithSampler` is not per-call on Chat — configure the sampler with `WithDefaultSampler` on `Client` before opening the chat.

## Branching Conversations (`Chat.Clone`)

Use `Chat.Clone()` to fork an active conversation in O(1) without re-prefilling its system prompt, tool definitions, or earlier turns. Each cloned `*Chat` carries its own KV cache and can diverge independently; keep the parent `*Chat` open until all clones finish and call `Close()` on each clone:

```go
chat, err := client.NewChat(ctx, litertlm.WithSystemPrompt("You are concise."))
defer chat.Close()

_, _ = chat.Send(ctx, "Context: Go was designed at Google in 2007.")

branch, err := chat.Clone()
if err != nil {
    return err
}
defer branch.Close()

r1, _ := chat.Send(ctx, "Who created Go?")
r2, _ := branch.Send(ctx, "What year did design start?")
```

See [`examples/clone/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/clone).

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
decode breakdown and reported KV-cache capacity (`MaxNumTokens`), read
`Conversation.BenchmarkInfo()` instead (requires `WithBenchmarkEnabled()`).

## See also

- [Tools guide](tools.md) — full reference for `RawTool`,
  `ManagedTool`, `RegisterTool`, schema reflection, the dispatch
  loop, `WithMaxToolHops`, `WithToolChoice`, and `ToolPolicy`.
- [`examples/chat/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/chat)
  — minimal multi-turn demo.
- [`examples/clone/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/clone)
  — KV-cache branching with `Chat.Clone`.
- [`examples/thinking/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/thinking)
  — per-chat and per-turn reasoning controls.
- [`examples/constrained-json/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/constrained-json)
  — grammar-constrained JSON Schema and regex output.
- [`examples/logit-penalties/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/logit-penalties)
  — repetition penalties, n-gram blocking, and token suppression.
- [`examples/tool-choice/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/tool-choice)
  — OpenAI-compatible `ToolChoice` policies (C API `1.1+`).
- [`examples/autotool/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/autotool)
  — typed tool registration with auto-dispatch.
- [`examples/conversation/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/conversation)
  — manual dispatch with `NewRawTool` + `SendToolResult`.
- [Structured output](structured-output.md) — when you want
  type-safe JSON instead of free-form text.
- [Low-level API](low-level.md) — `Conversation`, `ConversationConfig`.
