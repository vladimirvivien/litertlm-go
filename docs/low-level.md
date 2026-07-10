# Low-level API

`litertlm-go` exports every C-API symbol as a Go method providing
fine-grained low-level control of the runtime engine.

Common reasons to drop down:

- **Manual prefill / decode.** Explicit `RunPrefill` / `RunDecode`
  sequencing in place of `Session.GenerateContent`.
- **Text scoring.** `Session.ScoreTexts` returns log-probabilities
  for candidate completions.
- **Token introspection.** `Engine.Tokenize` / `Engine.Detokenize`,
  `Engine.StartTokenIDs` / `StopTokenIDs`.
- **Deterministic resource lifetimes.** Explicit `Delete()` on each
  C-backed handle.
- **Multimodal inputs.** `[]InputData` with
  `NewBinaryInput(InputImage, …)`. For ergonomic high-level
  multimodal calls, use `Client.GenerateMulti` /
  `GenerateDataMulti[T]` with `Part` constructors — see
  [Client → Multimodal inputs](client.md#multimodal-inputs).
- **Custom Conversation flows.** Sequencing tool-call and
  tool-response messages directly, prefilled message histories, etc.

## Method map

| High-level                            | Low-level equivalent                                                                                                       |
|---------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| `litertlm.New(ctx, opts...)`          | `Load` + `NewEngineSettings` / `NewEngineSettingsFromFd` + setters + `NewEngine`                                           |
| `client.Close()`                      | `engine.Delete()` + `settings.Delete()`                                                                                    |
| `client.Generate(ctx, prompt)`        | `engine.NewSession(0)` + `session.GenerateContent([]InputData{NewTextInputString(prompt)})` + `resp.Text(0)` + `Delete`s   |
| `client.GenerateStream(ctx, prompt)`  | `session.GenerateContentStreamCh(...)` channel                                                                             |
| `client.GenerateResponse(ctx, prompt)`| `session.GenerateContent(...)` returning `Responses` directly                                                              |
| `client.NewChat(ctx, opts...)`        | `NewConversationConfig` + `engine.NewConversation`                                                                         |
| `chat.Send(ctx, msg)`                 | `conv.SendMessage(messageJSON, "", OptionalArgs(0))` + parse JSON envelope                                                  |
| `chat.SendStream(ctx, msg)`           | `conv.SendMessageStreamCh(messageJSON, "", OptionalArgs(0))` channel                                                        |
| `chat.SendToolResult(name, result)`   | Build `{"role":"tool","content":[{"name":..., "response":...}]}` JSON, then `conv.SendMessage`                              |
| `chat.Close()`                        | `conv.Delete()` + `convCfg.Delete()`                                                                                       |
| `litertlm.GenerateData[T]`            | `Generate` + manual prompt augmentation + JSON extraction + `json.Unmarshal`                                               |

## Resource lifetime management

The low-level API surfaces every C handle as a `uintptr` value type
(`Engine`, `Session`, `EngineSettings`, `Conversation`, `Responses`,
`BenchmarkInfo`, `JsonResponse`, `InputData`, …). Each has a `.Delete()` method.

**Rules of thumb:**

- Every `New*` and every `Generate*` (when it returns a handle) must
  be paired with `defer h.Delete()`.
- Strings returned by accessor methods are copied into Go memory, so
  they remain valid after the parent handle is deleted.

## Examples
### Explicit prefill → decode

```go
session, _ := engine.NewSession(0)
defer session.Delete()

if err := session.RunPrefill([]litertlm.InputData{
    litertlm.NewTextInputString("The capital of France is"),
}); err != nil {
    return err
}

resp, err := session.RunDecode()
if err != nil {
    return err
}
defer resp.Delete()

fmt.Println(resp.Text(0))
```

See [`examples/prefill-decode/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/prefill-decode).

### Text scoring

```go
session, _ := engine.NewSession(0)
defer session.Delete()

session.RunPrefill([]litertlm.InputData{litertlm.NewTextInputString(prompt)})

resp, _ := session.ScoreTexts([]string{" Paris."}, true)
defer resp.Delete()

score, _ := resp.Score(0)
length, _ := resp.TokenLength(0)
fmt.Printf("score=%v length=%d\n", score, length)
```

See [`examples/score/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/score).

### Tokenize

```go
tokens, _ := engine.Tokenize("Hello, world.")
roundTrip, _ := engine.Detokenize(tokens)
fmt.Println(tokens)
fmt.Println(roundTrip)
```

See [`examples/tokenize/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/tokenize).
