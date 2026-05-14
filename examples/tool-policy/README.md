# tool-policy

Contrast the two values of `WithToolPolicy` by running a tool whose
handler errors on inputs missing from its lookup table.

## What this exercises

`WithToolPolicy` is a per-tool option passed to `RegisterTool`. It
governs how a handler error is propagated when the model calls the
tool during auto-dispatch.

| Policy | Handler error behavior |
|---|---|
| `ToolPolicyReturnOnError` (default) | The error is wrapped as `litertlm: tool %q: %w` and returned from `Chat.Send`. The dispatch loop ends. The model never sees the error. |
| `ToolPolicyInformOnError` | The error message is marshaled as `{"error": "..."}` and sent back as the tool's result. The dispatch loop continues; the model can apologize, retry with different arguments, or pick another tool. |

`WithToolPolicy` is *not* a chat-level Auto / Required / Disabled
toggle — no such API exists in the wrapper today.

## Prerequisites

- A `.litertlm` model file. Gemma 4 family with tool-calling support
  works.
- LiteRT-LM shared libraries on disk. Pass `-lib <dir>` or set
  `LITERTLM_LIB=<dir>`.

## Flags

| Flag | Default | Description |
|---|---|---|
| `-model` | (required) | Path to `.litertlm` model file. |
| `-lib` | `$LITERTLM_LIB` | Directory holding the shared libraries. |
| `-backend` | `cpu` | `cpu` or `gpu`. |
| `-prompt` | weather-in-Atlantis prompt | User message. |
| `-policy` | (required) | `return` or `inform`. |

`-policy` is required because `RegisterTool` rejects duplicate tool
names and the wrapper has no deregister API. Each policy is therefore
exercised by a separate binary invocation.

## Run

Run twice, once with each policy, to see the contrast:

```sh
go run ./examples/tool-policy -model "$MODEL" -policy return
go run ./examples/tool-policy -model "$MODEL" -policy inform
```

## Expected output

### `-policy return`

```
policy>    return
user>      What's the weather in Atlantis? If that lookup fails, try Tokyo.
(invoke)   lookup_weather(city="Atlantis")
send: litertlm: tool "lookup_weather": no weather data for city "Atlantis"
exit status 1
```

The handler error is wrapped and returned. The model is given no
chance to recover; the conversation terminates after a single hop.

### `-policy inform`

```
policy>    inform
user>      What's the weather in Atlantis? If that lookup fails, try Tokyo.
(invoke)   lookup_weather(city="Atlantis")
assistant> I couldn't find the weather for Atlantis. Would you like me to check the weather in Tokyo instead?
```

The handler error is marshaled as the tool's result back to the
model, so the dispatch loop continues. What the model does next is
its own choice — depending on sampling and prompt, it may retry the
tool with a different argument, pick a different tool, or surface
the failure to the user as above. The contrast with `-policy
return` is that the model gets a *chance* to react.

## Lookup table

The tool consults an in-process map of four cities (Paris, Tokyo,
London, Lagos). Cities outside that set return an error from the
handler — that is the path the example uses to trigger the policy
branch.

## Notes

- Both policies obey `WithMaxToolHops` (default 5). Under
  `ToolPolicyInformOnError`, a model that keeps requesting
  unsupported cities will eventually hit the cap and `Chat.Send`
  returns `ErrToolHopsExceeded`.
- The model receives exactly `{"error": err.Error()}` — no extra
  fields. Handlers wanting to give the model structured retry hints
  should return a typed value with an `error` (or similar) field
  rather than a Go `error`.
- A single chat may host tools registered with mixed policies; the
  policy applies per tool, not per chat.
