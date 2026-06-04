# litertlm-go examples

This directory contains Go programs demonstrating the use of the `litertlm-go` API.
Every example takes the same baseline flags:

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
go run ./examples/<name> -model /abs/path/to/<model>.litertlm [-backend cpu|gpu] [-lib=$LITERLM_LIB]
```

`-backend` defaults to `cpu` and `-lib` defaults to `$LITERTLM_LIB`.

## Index

| Path                          | What it shows                                                      |
|-------------------------------|--------------------------------------------------------------------|
| `examples/hello/`             | Minimal `Generate`                                                 |
| `examples/stream/`            | `GenerateStream` with range-over-func                              |
| `examples/chat/`              | Multi-turn `Chat` with a system prompt                             |
| `examples/chat-history/`      | Seed `Chat` with a prior transcript via `WithInitialMessages` + `WithExtraContext` / `WithFilterChannelContentFromKVCache` / `WithMaxToolHops` |
| `examples/bot/`               | Persistent multimodal chatbot — slash commands, `MEM.log` history, image / audio attachments via `-attach`, replay through `WithInitialMessages` on restart |
| `examples/conversation/`      | `Chat` + `NewRawTool` + manual dispatch via `Reply.ToolCalls`      |
| `examples/autotool/`          | `Chat` + `RegisterTool` + auto-dispatch                            |
| `examples/tool-policy/`       | `WithToolPolicy(ToolPolicyReturnOnError vs ToolPolicyInformOnError)` — handler-error behavior under auto-dispatch |
| `examples/clone/`             | `Chat.Clone` — branch a prefilled Chat into independent conversations that share KV state |
| `examples/structured/`        | `GenerateData[T]` (typed JSON output via reflection)               |
| `examples/vision/`            | `GenerateMulti` (image + text) with self-comparison against a sidecar |
| `examples/audio/`             | `GenerateMulti` (audio + text) — transcription with optional alignment vs a reference |
| `examples/extract/`           | `GenerateDataMulti[T]` (image-to-typed-JSON) with self-comparison  |
| `examples/cancel/`            | Cancelling a streaming generation via `context.WithCancel`         |
| `examples/prefill-decode/`    | Explicit two-phase generation (low-level)                          |
| `examples/conversation-lowlevel/` | Low-level twin of Chat: hand-built `SessionConfig` + `ConversationConfig` + `SendMessage` + `RenderMessage` + `BenchmarkInfo` |
| `examples/score/`             | `ScoreTexts` + `Score` / `TokenLength` (low-level)                 |
| `examples/token-scores/`      | `ScoreTexts` + `TokenScores` per-token log-probs paired with `Engine.Tokenize` |
| `examples/raw-multi/`         | `GenerateMulti` / `GenerateMultiStream` / `GenerateMultiResponse` — three call shapes for the same image + text input |
| `examples/tokenize/`          | `Client.Tokenize` / `Client.TokenLength` + `Engine.Detokenize` / start / stop tokens via `Client.Engine()` |
| `examples/token-count/`       | `Chat.TokenCount` — running KV-cache token count per turn (benchmark-free), projected against `WithMaxTokens` |
| `examples/gpu/`               | GPU-backed generation                                              |
| `examples/benchmarks/`        | `Response.Benchmark()` (high-level) vs `Session.BenchmarkInfo()` (low-level) side-by-side |
| `examples/cache-warmup/`      | Cold-vs-warm `WithCacheDir` load — XNNPACK / mldrift artefact reuse        |
| `examples/activation-dtype/`  | Default-vs-selected `WithActivationDataType` (F32 / F16 / I16 / I8) — empirical per-backend deltas |
| `examples/prefill-chunk/`     | Default-vs-selected `WithPrefillChunkSize` (CPU-only) — chunked vs unchunked prefill timings |
| `examples/parallel-load/`     | Parallel vs serial `WithParallelSectionLoading` — `litertlm.New` wall-clock delta |
| `examples/logging/`           | `SetMinLogLevel` — set the LiteRT-LM log severity floor at startup and toggle mid-program |
| `examples/per-call-sampler/`  | `WithSampler` per-call override — three sampler shapes (Deterministic / Balanced / Creative) on the same Client |
| `examples/speculative/`       | Side-by-side throughput comparison with / without `WithSpeculativeDecodingEnabled` |
