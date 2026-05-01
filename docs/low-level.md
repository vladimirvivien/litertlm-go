# Low-level API

The package exports every C-API symbol as a Go method. The high-level
`Client` / `Chat` types are wrappers built on top — when their
ergonomics don't fit your use case, drop down.

Common reasons to drop down:

- **Manual prefill / decode.** Run prefill, do something between, run
  decode. The high-level `Generate` always does both back-to-back.
- **Text scoring.** `Session.ScoreTexts` returns log-probabilities for
  candidate completions. No high-level wrapper today.
- **Token introspection.** `Engine.Tokenize` / `Engine.Detokenize`,
  `Engine.StartTokenIDs` / `StopTokenIDs`. All exposed at low level.
- **Deterministic resource lifetimes.** High-level `Response` /
  `Reply` use `runtime.AddCleanup`; if you're memory-bound and want
  explicit `.Delete()` after each call, low-level is the path.
- **Multimodal inputs.** `[]InputData` with `NewBinaryInput(InputImage, …)`
  for image / audio bytes. The high-level API is text-only.
- **Custom Conversation flows.** Sequencing tool-call and tool-response
  messages by hand, prefilled message histories, etc.

## Method map

| High-level                            | Low-level equivalent                                                                                                       |
|---------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| `litertlm.New(ctx, opts...)`          | `Load` + `NewEngineSettings` + setters + `NewEngine`                                                                       |
| `client.Close()`                      | `engine.Delete()` + `settings.Delete()`                                                                                    |
| `client.Generate(ctx, prompt)`        | `engine.NewSession(0)` + `session.GenerateContent([]InputData{NewTextInputString(prompt)})` + `resp.Text(0)` + `Delete`s   |
| `client.GenerateStream(ctx, prompt)`  | `session.GenerateContentStreamCh(...)` channel                                                                             |
| `client.GenerateResponse(ctx, prompt)`| `session.GenerateContent(...)` returning `Responses` directly                                                              |
| `client.NewChat(ctx, opts...)`        | `NewConversationConfig` + `engine.NewConversation`                                                                         |
| `chat.Send(ctx, msg)`                 | `conv.SendMessage(messageJSON, "")` + parse JSON envelope                                                                  |
| `chat.SendStream(ctx, msg)`           | `conv.SendMessageStreamCh(...)` channel                                                                                    |
| `chat.SendToolResult(name, result)`   | Build `{"role":"tool","content":[{"name":..., "response":...}]}` JSON, then `conv.SendMessage`                              |
| `chat.Close()`                        | `conv.Delete()` + `convCfg.Delete()`                                                                                       |
| `litertlm.GenerateData[T]`            | `Generate` + manual prompt augmentation + JSON extraction + `json.Unmarshal`                                               |

## Lifetime rules

The low-level API surfaces every C handle as a `uintptr` value type
(`Engine`, `Session`, `EngineSettings`, `Conversation`, `Responses`,
`BenchmarkInfo`, `JsonResponse`, …). Each has a `.Delete()` method.

**Rules of thumb:**

- Every `New*` and every `Generate*` (when it returns a handle) must
  be paired with `.Delete()` once you're done.
- `defer h.Delete()` immediately after a successful constructor is
  the easiest way to keep this right.
- Calling `.Delete()` on a zero-valued handle (`var h Session`) is a
  no-op. Safe.
- Strings returned by accessor methods are copied into Go memory, so
  they remain valid after the parent handle is deleted.

The high-level `*Response` and `*Reply` types use `runtime.AddCleanup`
instead of explicit `Delete` for ergonomics. If you mix high-level
and low-level — say, calling `client.Engine().NewSession(0)` to get a
raw `Session` — the rules apply only to the handles you obtain
directly from the low-level API.

## Receiver style

Every handle type uses **value receivers**:

```go
func (s Session) GenerateContent(...) (Responses, error)
```

That's because the value already *is* the C pointer (a uintptr); a
pointer receiver would add a level of indirection without benefit.
The FFI plumbing also depends on it — `unsafe.Pointer(&s)` is
`*uintptr`, which is what the C side expects. Pointer receivers
would break the call-site uniformity used throughout the bindings.

## Concurrency

The C-side thread-safety contract isn't documented. The high-level
`Client` serialises `engine.NewSession` under a mutex out of caution.
At the low level, you're on your own — don't share a `Session`
between goroutines, don't call `engine.NewSession` from two
goroutines without your own synchronisation.

`runtime.KeepAlive` is used in a handful of places where the C side
reads through an `unsafe.Pointer` after the Go function would
otherwise be free to drop the variable. Don't remove those calls.
The kept-alive site has a short comment explaining the contract.

## Example: explicit prefill → decode

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

## Example: text scoring

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
The CPU engine currently rejects `len(targets) > 1`; the API is
forward-compatible for when upstream relaxes that.

## Example: tokenize

```go
tokens, _ := engine.Tokenize("Hello, world.")
roundTrip, _ := engine.Detokenize(tokens)
fmt.Println(tokens)
fmt.Println(roundTrip)
```

See [`examples/tokenize/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/tokenize).

## See also

- [`pkg.go.dev` reference](https://pkg.go.dev/github.com/vladimirvivien/litertlm-go/pkg/litertlm)
  — godoc for every exported symbol.
- [Troubleshooting](troubleshooting.md) — quirks worth knowing
  about, especially when working at the low level.
