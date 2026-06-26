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

When you register a Go function as a tool, the library automatically translates the input struct into a JSON Schema. The rules are simple:

* **Exported Fields**: Only exported fields are included in the schema.
* **Names & Descriptions**: Field names default to the Go field name (lowercased) or are customized using the `json` tag. You can add descriptions using the `description` tag.
* **Required vs. Optional**: Non-pointer fields are required; pointer fields are optional.
* **Supported Types**: Supports strings, booleans, integers, floats, slices, arrays, nested structs, and pointers.

The input type `I` must be a struct or a pointer-to-struct. Other types will return an error during registration.

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

### Per-call dispatch knobs

Three `RuntimeOption` values control dispatch on a per-`Send*` basis
(see [Chat → Per-call options](chat.md#per-call-options) for the full
list):

- `WithMaxConcurrentTools(n)` — when a reply contains multiple tool
  calls, dispatch handlers concurrently capped at `n` in-flight.
  `n <= 1` (the default) is sequential. Result ordering in the
  follow-up tool-role message matches the model's call order
  regardless of completion order. Handlers must be safe to invoke
  from multiple goroutines when `n > 1`.
- `WithReturnToolRequests(true)` — bypass dispatch entirely for the
  current `Send` / `SendMulti` / `SendToolResult`. The first reply
  carrying tool calls is returned via `Reply.ToolCalls()` for manual
  handling. Streaming methods ignore the flag.
- `WithMaxToolHops(n)` — see below; applies at `NewChat` time, not
  per call.

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

`SendStream` and `SendMultiStream` run the same dispatch loop as
`Send` / `SendMulti`. Text chunks from each turn are yielded
immediately; intermediate `Final` markers from inner turns are
suppressed. The iterator emits one trailing synthetic `Final` chunk
after the loop terminates. Replies whose tool calls include any
non-dispatchable entry (a `RawTool` or an unknown name) end the
stream with an error.

Streaming methods ignore `WithReturnToolRequests` — manual handling
of tool calls requires a `*Reply`, so use `Send` for that flow.

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
