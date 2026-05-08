# Tools

Tools let the model trigger function calls — fetching weather, doing
arithmetic, looking up records — and incorporate the results back into
its reply. `litertlm-go` offers two flavors:

| Flavor        | Constructor    | Dispatch                                              |
|---------------|----------------|-------------------------------------------------------|
| `RawTool`     | `NewRawTool`   | Manual — `Reply.ToolCalls()` + `Chat.SendToolResult`  |
| `ManagedTool` | `RegisterTool` | Framework dispatches the typed handler                |

Both implement `ToolDefinition` and may be mixed in the same
`WithTool(...)` call.

```go
type ToolDefinition interface {
    Name() string
    Description() string
    Parameters() map[string]any
}
```

## `RegisterTool` — typed tools with auto-dispatch

`RegisterTool` reflects over the input struct to build a JSON-Schema
parameters object, registers the tool on the `Client` so its name
cannot be reused, and returns a `*ManagedTool[I, O]` ready to attach
to chats with `WithTool`.

```go
type WeatherInput struct {
    Location string `description:"city and state, e.g. 'Boston, MA'"`
}
type WeatherOutput struct {
    Forecast string `json:"forecast"`
}

weather, err := litertlm.RegisterTool(client, "get_weather",
    "Get a weather forecast for a US city",
    func(ctx context.Context, in WeatherInput) (WeatherOutput, error) {
        return WeatherOutput{Forecast: forecastFor(in.Location)}, nil
    })
```

### Schema reflection rules

For the input struct `I`:

- Exported fields only.
- Field name from `json:"name"` tag, else lowercased Go name.
  `json:"-"` excludes the field entirely.
- Description from `description:"..."` tag, optional.
- Pointer fields are optional; non-pointer fields are required.
- Supported kinds: `string`, `bool`, every `int*`/`uint*`/`float*`
  width, slice / array, nested struct, pointer (unwrapped).
- Recursion capped at depth 32.

`I` must be a struct or pointer-to-struct. Other kinds return an
error from `RegisterTool`.

### Generated schema shape

The resulting parameters map matches the OpenAI / Anthropic
function-calling convention:

```json
{
    "type": "object",
    "properties": {
        "location": {"type": "string", "description": "city and state, e.g. 'Boston, MA'"}
    },
    "required": ["location"]
}
```

## `NewRawTool` — hand-built declarations

For declarations that don't fit the reflection rules (dynamic
parameters, hand-crafted schemas, tools whose handlers live outside
the chat lifecycle), construct the parameters map yourself:

```go
search := litertlm.NewRawTool("search_web",
    "Search the public web",
    map[string]any{
        "type": "object",
        "properties": map[string]any{
            "query": map[string]any{"type": "string"},
        },
        "required": []string{"query"},
    })
```

`RawTool` carries a name, description, and parameters but no handler.
The model can call it; the caller handles the call manually with
`Reply.ToolCalls()` and `Chat.SendToolResult`.

## Attaching tools to a chat

`WithTool` is variadic and accepts any `ToolDefinition`:

```go
chat, _ := client.NewChat(ctx,
    litertlm.WithSystemPrompt("..."),
    litertlm.WithTool(weather, search, calcAdd),
)
defer chat.Close()
```

Multiple `WithTool(...)` calls accumulate. Duplicate names error at
`NewChat` time.

## Auto-dispatch loop

When the chat has at least one `ManagedTool` registered, `Chat.Send`
runs a dispatch loop:

```
Send(user message)
        │
        ▼
   transport.SendMessage  ──►  reply
                                  │
                                  ▼
              ┌── reply has no tool calls? ──► return reply ✓
              │
              └── every call dispatchable?
                       │
            ┌──────────┴──────────┐
            │                     │
           yes                    no
            │                     │
            ▼                     ▼
   invoke each handler    return reply for manual handling
            │
            ▼
   bundle results, loop ──► next reply ──► (back to top)
```

A call is **dispatchable** when its name maps to a `ManagedTool` in
the chat's registry. `RawTool` entries and unknown names are
non-dispatchable; if any call in a reply is non-dispatchable, the
loop returns the reply for the caller to handle the whole turn
manually.

`Chat.SendToolResult` enters the same loop — the model's response to
a tool result might trigger more calls, which the framework dispatches
the same way.

### `WithMaxToolHops(n)` — cap iterations

Each iteration is one model→tool→model round-trip. The default cap is
**5**; override per chat:

```go
chat, _ := client.NewChat(ctx,
    litertlm.WithTool(weather),
    litertlm.WithMaxToolHops(3),
)
```

Exceeding the cap returns an error matching
`errors.Is(err, ErrToolHopsExceeded)`. Use `errors.As` to a
`*ToolHopsError` for the partial last reply:

```go
reply, err := chat.Send(ctx, prompt)
if err != nil {
    var hops *litertlm.ToolHopsError
    if errors.As(err, &hops) {
        log.Printf("model still calling tools after %d hops", hops.Hops)
        log.Printf("last reply: %s", hops.LastReply.Raw())
    }
    return err
}
```

`n <= 0` is ignored (default applies). The cap exists to bound a
runaway model, not to disable dispatch — chats with no `ManagedTool`
registered never enter the loop.

## `ToolPolicy` — error handling per tool

Each `ManagedTool` carries a `ToolPolicy` set with `WithToolPolicy`
at registration time. The policy controls what happens when the
handler returns an error during dispatch:

| Policy                          | Behavior                                                                    |
|---------------------------------|-----------------------------------------------------------------------------|
| `ToolPolicyReturnOnError` (default) | Propagate the error from `Chat.Send`; loop ends. The model is not informed. |
| `ToolPolicyInformOnError`       | Marshal `{"error": "<err.Error()>"}` as the tool's response and continue. The model can retry, apologize, or pick a different tool. |

```go
// Default: a validation error stops the loop and surfaces to the caller.
validate, _ := litertlm.RegisterTool(client, "validate_input", "...",
    func(ctx context.Context, in ValidateInput) (Validated, error) {
        return doValidate(in)
    })

// Inform-on-error: transient failures the model can react to.
weather, _ := litertlm.RegisterTool(client, "get_weather", "...",
    func(ctx context.Context, in WeatherInput) (WeatherOutput, error) {
        return forecastFor(in.Location)
    },
    litertlm.WithToolPolicy(litertlm.ToolPolicyInformOnError))
```

Pick `ToolPolicyReturnOnError` for invariants the framework should
defend (a corrupted state is a programmer bug; bubble it up). Pick
`ToolPolicyInformOnError` for transient failures the model can
reasonably handle conversationally (rate limits, retryable network
errors, ambiguous queries the model might rephrase).

## Streaming

`SendStream` does not wrap the dispatch loop. It returns the raw
streaming chunks of the model's first reply. For typed-tool flows
that need streaming, call `Send` and stream the post-dispatch reply
yourself.

## See also

- [`examples/autotool/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/autotool)
  — auto-dispatch with a typed handler.
- [`examples/conversation/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/conversation)
  — manual dispatch with `NewRawTool` + `SendToolResult`.
- [Chat](chat.md) — the multi-turn `Chat` API; tool registration is
  one of its options.
- [Structured output](structured-output.md) — `GenerateData[T]` and
  `GenerateDataMulti[T]` for one-shot structured replies without the
  multi-turn machinery.
