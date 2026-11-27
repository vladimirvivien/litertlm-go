# litertlm-go examples

This directory contains Go programs demonstrating the use of the `litertlm-go` API.
Every example supports both local paths and automated downloads:

```bash
# Automated provisioning (downloads libraries and model directly):
go run ./examples/<name> -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4

# Or using pre-existing local paths / environment variables:
LITERTLM_LIB=/abs/path/to/dist/lib \
LITERTLM_MODEL=/abs/path/to/<model>.litertlm \
go run ./examples/<name> [-backend cpu|gpu]
```

### Common Flags

* `-get-lib <version>`: Automatically download and cache LiteRT-LM shared libraries (e.g. `v0.19.1`).
* `-get-model <id/url>`: Automatically download and cache a `.litertlm` model from Hugging Face or direct URL (e.g. `litert-community/gemma3-1b-it-int4`).
* `-model <path>`: Path to a local `.litertlm` model file (falls back to `LITERTLM_MODEL` env).
* `-lib <path>`: Path to the LiteRT-LM shared library directory (falls back to `LITERTLM_LIB` env).
* `-backend <cpu|gpu>`: Execution backend (defaults to `cpu`).

## Index

| Path                          | What it shows                                                      |
|-------------------------------|--------------------------------------------------------------------|
| `examples/capabilities/`      | `InspectModel` — zero-allocation model container type, context limits, backend/NPU metadata, and embedding dimensions |
| `examples/hello/`             | Minimal `Generate`                                                 |
| `examples/stream/`            | `GenerateStream` with range-over-func                              |
| `examples/embed/`             | Standalone text, multimodal, batch, and Matryoshka-truncated embeddings with `EmbeddingClient` |
| `examples/semantic-search/`   | Modular `search` and zero-shot `classify` packages + CLI over mixed text and images |
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
| `examples/speculative/`       | Three-way throughput comparison (`Off` / `On` / `Adaptive` + `-lenience`) with `WithSpeculativeDecoding` |
| `examples/checkpoints/`       | KV-cache branching with `Session.SaveCheckpoint`, `RewindToCheckpoint`, `Session.Clone`, and `RunDecodeAsyncCh` |
| `examples/constrained-json/`  | Grammar-enforced `ConstraintJSONSchema` and `ConstraintRegex` generation via `WithConstraintProvider` / `WithConstraint` |
| `examples/thinking/`          | Session and per-turn reasoning budgets with `WithThinking`, `WithTurnThinking`, and KV channel filtering |
| `examples/logit-penalties/`   | Controlling repetition loops and token bans with `WithRepetitionPenalty`, `WithNoRepeatNgram`, and `WithSuppressTokens` |
| `examples/tool-choice/`       | Per-turn OpenAI `ToolChoice` policies (`auto`, `none`, `required`, forced function, `allowed_tools`, C API 1.1.0+) |
| `examples/template-inspector/` | Template dry-run and raw prompt template introspection via `Chat.Conversation().RenderPreface()` |
| `examples/fd-loader/`          | Sandboxed model loading directly from open file descriptors (`WithModelFD`) |
| `examples/thread-tuner/`       | Benchmark profiling of CPU thread allocation sizes (`WithNumThreads`) |
| `examples/token-limiter/`      | Dynamic sequence output constraints on a per-call basis (`WithMaxOutputTokens`) |
| `examples/lora-tuner/`         | Low-level base and supported LoRA rank configuration on settings handles at initialization |
| `examples/model-downloader/`   | Standalone model downloader utility using `litertlm.FetchModel` |
| `examples/model-sync/`         | Differential `.litertlm` update planning and cache sync via `PlanModelSync` / `SyncModelCache` (C API `1.1.0+`) |
| `examples/metal-residency/`    | Apple Metal residency startup option and runtime toggle via `Client.SetMetalResidency` |
| `examples/debugger/`           | AI Edge Debugger configuration (`WithDebuggerEnabled`, C API `1.1.0+`) and `DebugCaptureDir()` inspection |
| `examples/config-file/`        | Configure `litertlm.Client` from a centralized `config.json` file using `WithConfigFile` |

