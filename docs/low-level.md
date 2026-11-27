# Low-level API

`litertlm-go` exports every C-API symbol as a Go method providing
fine-grained low-level control of the runtime engine.

Common reasons to drop down:

- **Manual prefill / decode & async streaming.** Explicit `RunPrefill` / `RunDecode` or `RunDecodeAsync` / `RunDecodeAsyncCh` sequencing in place of `Session.GenerateContent`.
- **KV-cache checkpoints, rewinding & branching.** Branch multiple continuations from one shared prompt with `SaveCheckpoint`, `RewindToCheckpoint`, `RewindToStep`, and `Session.Clone` / `Conversation.Clone`.
- **Per-turn constraint, thinking & penalty handles.** Configure `OptionalArgs` with `SetThinkingConfig`, `SetConstraint`, `SetRepetitionPenalty`, `SetNoRepeatNgram`, `SetSuppressTokens`, and `SetToolChoice` (C API `1.1+`).
- **Model capability inspection.** Read `.litertlm` metadata (`ModelType`, `MaxContextTokens`, `IsDynamicContext`, `ModalitySupportedBackends`, NPU/SoC metadata, embedding dimensions) directly via `OpenLoadedFile` / `LoadedFile` without creating an engine.
- **Standalone embedding engines.** Use `EmbeddingEngineSettings`, `EmbeddingEngine`, `EmbeddingOptions`, `EmbeddingResponse`, and `EmbeddingResponses` directly when reusing `InputData` buffers or option handles across calls.
- **Differential model sync (`C API 1.1+`).** Drive `NewModelSyncOptions`, `NewModelSyncPlanFromFile`, `ModelSyncUpdate`, and `ModelSyncCacheSync` directly over a custom `RangeFetcher` or `ModelSyncRangeWriter`.
- **Runtime Metal residency & debugger captures.** Toggle residency with `Engine.SetMetalResidency`, configure `EngineSettings.SetEnableDebugger(true)` (`C API 1.1+`), and inspect trace folders via `Session.DebugInfo` / `Session.DebugCaptureDir` and `Conversation.SessionDebugInfo` / `Conversation.DebugCaptureDir`.
- **Single-threaded execution.** Drive caller-thread scheduling with `SetSingleThreadedExecution(true)` paired with `Conversation.WaitUntilDone`.
- **Text scoring.** `Session.ScoreTexts` returns log-probabilities for candidate completions.
- **Token introspection.** `Engine.Tokenize` / `Engine.Detokenize`, `Engine.StartTokenIDs` / `StopTokenIDs`.
- **Deterministic resource lifetimes.** Explicit `Delete()` on each C-backed handle, with error checks via `*CError`.
- **Multimodal inputs.** `[]InputData` with `NewBinaryInput(InputImage, …)`. For high-level multimodal calls, use `Client.GenerateMulti` / `GenerateDataMulti[T]` with `Part` constructors — see [Client → Multimodal inputs](client.md#multimodal-inputs).

## Method map

| High-level                            | Low-level equivalent                                                                                                       |
|---------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| `litertlm.New(ctx, opts...)`          | `Load` + `NewEngineSettings` / `NewEngineSettingsFromFD` + setters + `NewEngine`                                           |
| `litertlm.InspectModel(path)`         | `Load` + `OpenLoadedFile` + `LoadedFile.Snapshot` + `LoadedFile.Delete`                                                    |
| `client.Close()`                      | `engine.Delete()` + `settings.Delete()`                                                                                    |
| `client.Generate(ctx, prompt)`        | `NewConversationConfig` + `engine.NewConversation` + `conv.SendMessage(msgJSON, "", OptionalArgs(0))` + `Delete`s          |
| `client.GenerateStream(ctx, prompt)`  | `conv.SendMessageStreamCh(...)` (or raw `session.GenerateContentStreamCh(...)` without chat templating)                    |
| `client.GenerateResponse(ctx, prompt)`| `conv.SendMessage(...)` + `conv.BenchmarkInfo()` (or raw `session.GenerateContent(...)` returning `Responses`)             |
| `client.NewChat(ctx, opts...)`        | `NewConversationConfig` + `engine.NewConversation`                                                                         |
| `chat.Clone()`                        | `conv.Clone()`                                                                                                             |
| `chat.Send(ctx, msg, opts...)`        | Build `OptionalArgs` + `conv.SendMessage(messageJSON, "", optArgs)` + parse JSON envelope                                  |
| `chat.SendStream(ctx, msg, opts...)`  | Build `OptionalArgs` + `conv.SendMessageStreamCh(messageJSON, "", optArgs)` channel                                        |
| `chat.SendToolResult(name, result)`   | Build `{"role":"tool","content":[{"name":..., "response":...}]}` JSON, then `conv.SendMessage`                              |
| `chat.Close()`                        | `conv.Delete()` + `convCfg.Delete()`                                                                                       |
| `litertlm.GenerateData[T]`            | Synthesized tool-call round-trip + schema-hinted `Generate` fallback + `json.Unmarshal`                                    |
| `litertlm.NewEmbeddingClient(...)`    | `Load` + `NewEmbeddingEngineSettings` + setters + `NewEmbeddingEngine`                                                     |
| `embedClient.Embed(ctx, text, ...)`   | `NewTextInputString` + `NewEmbeddingOptions` + `embeddingEngine.ComputeEmbedding` + `Values` + `Delete`s                   |
| `embedClient.EmbedBatch(ctx, ...)`    | Build `[][]InputData` + `embeddingEngine.ComputeEmbeddingBatch` + `EmbeddingResponses.Values` + `Delete`s                  |
| `client.SetMetalResidency(on)`        | `engine.SetMetalResidency(on)`                                                                                             |
| `chat.DebugCaptureDir()`              | `conv.SessionDebugInfo()` + `SessionDebugInfo.CaptureDir()` + `SessionDebugInfo.Delete()`                                  |
| `litertlm.PlanModelSync(...)`         | `NewModelSyncOptions` + `NewModelSyncPlanFromFile` + `ModelSyncPlanHandle.Snapshot` + `Delete`s                            |
| `litertlm.SyncModel(...)`             | `NewModelSyncOptions` + `ModelSyncUpdate` + `ModelSyncOptionsHandle.Delete`                                                |
| `litertlm.SyncModelCache(...)`        | `NewModelSyncOptions` + `ModelSyncCacheSync` + `ModelSyncResultHandle.CachedPath` + `Delete`s                              |

## Resource lifetime management

The low-level API surfaces each C handle as a `uintptr` value type (`Engine`, `Session`, `EngineSettings`, `SessionConfig`, `Conversation`, `ConversationConfig`, `OptionalArgs`, `Responses`, `BenchmarkInfo`, `JSONResponse`, `InputData`, `LoadedFile`, `EmbeddingEngine`, `EmbeddingEngineSettings`, `EmbeddingOptions`, `EmbeddingResponse`, `EmbeddingResponses`, `ModelSyncOptionsHandle`, `ModelSyncPlanHandle`, `ModelSyncResultHandle`, `SessionDebugInfo`). Each provides a `.Delete()` method. `ModelSyncRangeWriter` is a borrowed handle valid only for the duration of a `RangeFetcher` callback and must not be retained. Low-level setters return `error`, wrapping C status failures in `*CError` with thread-local error messages and `StatusCode` sentinels.

**Rules of thumb:**

- Every `New*`, `OpenLoadedFile`, `Clone`, `ComputeEmbedding*`, `ModelSyncCacheSync`, non-zero `DebugInfo()` / `SessionDebugInfo()`, and `Generate*` (when returning a handle) must be paired with `defer h.Delete()`.
- `EmbeddingResponses.At(i)` returns a borrowed `EmbeddingResponse` owned by the batch container; do not call `Delete()` on it.
- A cloned `Session` or `Conversation` owns an independent handle, but keep parent resources (`Engine`, root `Chat`) alive for the lifetime of active clones.
- Strings and embedding float slices returned by accessor methods are copied into Go memory, so they remain valid after the parent handle is deleted.

## Examples
### Explicit prefill → decode

```go
session, err := engine.NewSession(0)
if err != nil {
    return err
}
defer session.Delete()

in, err := litertlm.NewTextInputString("The capital of France is")
if err != nil {
    return err
}
defer in.Delete()

if err := session.RunPrefill([]litertlm.InputData{in}); err != nil {
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

### KV-cache checkpoints, rewinding, and cloning

Prefill a shared prefix once, save a named checkpoint or clone the session, then run async decoding across diverging branches:

```go
shared, _ := litertlm.NewTextInputString("In a small seaside village,")
defer shared.Delete()
_ = session.RunPrefill([]litertlm.InputData{shared})
_ = session.SaveCheckpoint("after-prefix")

cloned, _ := session.Clone()
defer cloned.Delete()

for sc := range session.RunDecodeAsyncCh() {
    if sc.Err != nil { return sc.Err }
    fmt.Print(sc.Text)
}

_ = session.RewindToCheckpoint("after-prefix")
// RewindToStep(0) clears back to step 0 when resetting completely.
// Continue decoding on either session or cloned independently.
```

See [`examples/checkpoints/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/checkpoints).

### Text scoring

Scoring advances model state; save a checkpoint before `ScoreTexts` and call `RewindToCheckpoint` afterward if the session will continue prefill or decode:

```go
session, _ := engine.NewSession(0)
defer session.Delete()

in, _ := litertlm.NewTextInputString(prompt)
defer in.Delete()
_ = session.RunPrefill([]litertlm.InputData{in})
_ = session.SaveCheckpoint("before-score")

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
