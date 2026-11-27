# Examples Gallery

`litertlm-go` ships with **48 self-contained runnable Go programs** under `examples/`, covering every API surface from one-line prompt generation to low-level KV-cache checkpoints, differential model sync, and semantic search. Browse by capability below to find working code you can run immediately on Linux, macOS, or Windows.

## How to Run Any Example

Every example accepts the same discovery flags so you can switch seamlessly between automatic downloads and local paths on disk:

=== "Automated Download"

    ```bash
    # Downloads and caches official LiteRT-LM shared libraries and model weights
    go run ./examples/<name> -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4
    ```

=== "Local Paths (Linux / macOS)"

    ```bash
    LITERTLM_LIB=/abs/path/to/dist/lib \
    LITERTLM_MODEL=/abs/path/to/model.litertlm \
        go run ./examples/<name> [-backend cpu|gpu]
    ```

=== "Local Paths (Windows PowerShell)"

    ```powershell
    $Env:LITERTLM_LIB = "$Env:USERPROFILE\include\litertlm\lib"
    $Env:LITERTLM_MODEL = "$Env:USERPROFILE\models\gemma3-1b-it-int4.litertlm"
    go run ./examples/<name> [-backend cpu]
    ```

| Common Flag | Fallback Env | Purpose |
|---|---|---|
| `-get-lib <ver>` | — | Download and cache LiteRT-LM shared libraries (e.g. `v0.19.1`). |
| `-get-model <id>` | — | Download a `.litertlm` model from Hugging Face shorthand or URL. |
| `-lib <dir>` | `LITERTLM_LIB` | Directory containing local LiteRT-LM shared libraries. |
| `-model <path>` | `LITERTLM_MODEL` | Path to a local `.litertlm` model file. |
| `-backend <mode>` | — | Target backend (`cpu`, `gpu`, or `ynnpack`; defaults to `cpu`). |

---

## Quickstart, Provisioning & Inspection

Start here to run a prompt in minimal code, validate model metadata before allocating memory, or manage config-driven provisioning.

| Example | API Surface Demonstrated | Run It |
|---|---|---|
| [`hello`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/hello) | Minimal synchronous `Client.Generate` | `go run ./examples/hello -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`stream`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/stream) | Token-by-token streaming via `Chat.SendStream` and Go `iter.Seq2` | `go run ./examples/stream -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`capabilities`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/capabilities) | Zero-allocation `.litertlm` metadata inspection with `InspectModel` | `go run ./examples/capabilities -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`model-downloader`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/model-downloader) | Standalone `.litertlm` artifact fetcher via `litertlm.FetchModel` | `go run ./examples/model-downloader -get-model litert-community/gemma3-1b-it-int4` |
| [`model-sync`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/model-sync) | Differential `.litertlm` cache updates with `PlanModelSync` / `SyncModelCache` (`1.1+`) | `go run ./examples/model-sync -help` |
| [`config-file`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/config-file) | Centralized engine and sampler profiles loaded via `WithConfigFile` | `go run ./examples/config-file -help` |

---

## Multi-Turn Chat & Conversations

Maintain conversational state across turns, seed transcripts on startup, inspect prompt templates, or branch shared KV history.

| Example | API Surface Demonstrated | Run It |
|---|---|---|
| [`chat`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/chat) | Multi-turn interactive `Chat` with a system prompt | `go run ./examples/chat -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`chat-history`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/chat-history) | Replay transcripts via `WithInitialMessages`, `WithExtraContext`, and KV channel filters | `go run ./examples/chat-history -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`bot`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/bot) | Persistent REPL chatbot with slash commands, `MEM.log` compaction, and image/audio attachments | `go run ./examples/bot -get-lib v0.19.1 -get-model litert-community/gemma-4-E2B-it` |
| [`clone`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/clone) | O(1) conversation branching from prefilled state via `Chat.Clone` | `go run ./examples/clone -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`token-count`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/token-count) | Track live KV-cache usage across turns with `Chat.TokenCount` | `go run ./examples/token-count -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`template-inspector`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/template-inspector) | Render raw chat prompt prefaces with `Chat.Conversation().RenderPreface()` | `go run ./examples/template-inspector -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |

---

## Multimodal Vision & Audio

Combine text, image, and audio `Part` slices across one-shot generation, streaming replies, and typed struct extraction.

| Example | API Surface Demonstrated | Run It |
|---|---|---|
| [`vision`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/vision) | Image + text Q&A via `GenerateMulti` and reference comparison | `go run ./examples/vision -get-lib v0.19.1 -get-model litert-community/gemma-4-E2B-it -image ./examples/testdata/img1.png -description ./examples/testdata/img1.md` |
| [`audio`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/audio) | Audio transcription and analysis with `AudioFromFile` and `GenerateMulti` | `go run ./examples/audio -help` |
| [`extract`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/extract) | Extract typed Go structs straight from images using `GenerateDataMulti[T]` | `go run ./examples/extract -get-lib v0.19.1 -get-model litert-community/gemma-4-E2B-it -image ./examples/testdata/img1.png -description ./examples/testdata/img1.md` |
| [`raw-multi`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/raw-multi) | Compare `GenerateMulti`, `GenerateMultiStream`, and `GenerateMultiResponse` on the same input | `go run ./examples/raw-multi -help` |

---

## Embeddings & Semantic Search

Build local retrieval indexes and zero-shot classifiers powered by multimodal `EmbeddingClient` vectors.

| Example | API Surface Demonstrated | Run It |
|---|---|---|
| [`embed`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/embed) | Text, batch, optional multimodal, overflow handling, and Matryoshka truncation via `EmbeddingClient` | `go run ./examples/embed -get-lib v0.19.1 -get-model litert-community/embeddinggemma-2-text-vision-440m-litert-lm` |
| [`semantic-search`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/semantic-search) | Reusable `search` index and zero-shot `classify` packages plus interactive multimodal CLI | `go run ./examples/semantic-search search -get-lib v0.19.1 -get-model litert-community/embeddinggemma-2-text-vision-440m-litert-lm -dir ./examples/testdata -query "parked cars"` |

---

## Tools & Structured Output

Connect local models to Go functions, tune failure recovery, enforce per-turn `ToolChoice`, or guarantee JSON Schema syntax.

| Example | API Surface Demonstrated | Run It |
|---|---|---|
| [`autotool`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/autotool) | Typed Go function registration with `RegisterTool` and automatic dispatch loop | `go run ./examples/autotool -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`conversation`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/conversation) | Manual tool loop control using `NewRawTool`, `Reply.ToolCalls`, and `Chat.SendToolResult` | `go run ./examples/conversation -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`tool-policy`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/tool-policy) | Compare `ToolPolicyReturnOnError` vs `ToolPolicyInformOnError` handler failures | `go run ./examples/tool-policy -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4 -policy inform` |
| [`tool-choice`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/tool-choice) | Enforce `ToolChoiceFunction`, `ToolChoiceAllowedTools`, and `ToolChoiceNone` (`1.1+`) | `go run ./examples/tool-choice -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`structured`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/structured) | One-shot typed Go struct decoding with `GenerateData[T]` and retries | `go run ./examples/structured -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`constrained-json`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/constrained-json) | Decode-time `ConstraintJSONSchema` and `ConstraintRegex` enforcement with LLGuidance | `go run ./examples/constrained-json -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |

---

## Decoding Controls, Sampling & Reasoning

Tune sampling randomness, allocate per-turn thinking budgets, prevent repetition loops, cancel streams mid-flight, or enable speculative drafts.

| Example | API Surface Demonstrated | Run It |
|---|---|---|
| [`per-call-sampler`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/per-call-sampler) | Override `WithSampler` per call (Deterministic, Balanced, Creative) on a single `Client` | `go run ./examples/per-call-sampler -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`thinking`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/thinking) | Configure `WithThinking`, `WithTurnThinking`, and KV channel filtering on reasoning models | `go run ./examples/thinking -help` |
| [`logit-penalties`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/logit-penalties) | Curb repetition with `WithRepetitionPenalty`, `WithNoRepeatNgram`, and `WithSuppressTokens` | `go run ./examples/logit-penalties -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`token-limiter`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/token-limiter) | Cap response length per call via `WithMaxOutputTokens` | `go run ./examples/token-limiter -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`cancel`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/cancel) | Propagate `context.WithCancel` cleanly into streaming `Conversation.Cancel` | `go run ./examples/cancel -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`speculative`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/speculative) | Compare `SpeculativeDecodingOff`, `On`, and `Adaptive` throughput with `-lenience` | `go run ./examples/speculative -help` |

---

## Hardware Acceleration & Performance Tuning

Measure latency and tokens/second across CPU threads, prefill chunking, compiled artifact caches, activation precision, GPU execution, and Metal residency.

| Example | API Surface Demonstrated | Run It |
|---|---|---|
| [`gpu`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/gpu) | GPU-accelerated inference (Metal / WebGPU / Direct3D 12) | `go run ./examples/gpu -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`benchmarks`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/benchmarks) | Compare high-level `Response.Benchmark()` and low-level `Session.BenchmarkInfo()` metrics | `go run ./examples/benchmarks -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`cache-warmup`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/cache-warmup) | Measure cold-vs-warm startup speedups with `WithCacheDir` compiled weight caches | `go run ./examples/cache-warmup -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`activation-dtype`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/activation-dtype) | Compare `F32`, `F16`, `I16`, and `I8` execution via `WithActivationDataType` | `go run ./examples/activation-dtype -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`prefill-chunk`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/prefill-chunk) | Tune prompt prefill cache locality with CPU `WithPrefillChunkSize` | `go run ./examples/prefill-chunk -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`parallel-load`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/parallel-load) | Benchmark parallel vs serial section deserialization (`WithParallelSectionLoading`) | `go run ./examples/parallel-load -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`thread-tuner`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/thread-tuner) | Sweep CPU thread counts using `WithNumThreads` to find peak throughput | `go run ./examples/thread-tuner -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`metal-residency`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/metal-residency) | Toggle Apple Metal weight residency at startup and runtime via `Client.SetMetalResidency` | `go run ./examples/metal-residency -help` |

---

## Low-Level Engine, Checkpoints & Diagnostics

Inspect tokens, score candidate completions, rewind KV-cache steps, load sandboxed file descriptors, configure LoRA ranks, or collect AI Edge Debugger traces.

| Example | API Surface Demonstrated | Run It |
|---|---|---|
| [`prefill-decode`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/prefill-decode) | Explicit two-phase `Session.RunPrefill` and `Session.RunDecode` | `go run ./examples/prefill-decode -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`conversation-lowlevel`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/conversation-lowlevel) | Hand-built `SessionConfig`, `ConversationConfig`, `SendMessage`, and `RenderMessage` | `go run ./examples/conversation-lowlevel -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`checkpoints`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/checkpoints) | KV-cache branching with `SaveCheckpoint`, `RewindToCheckpoint`, `Session.Clone`, and `RunDecodeAsyncCh` | `go run ./examples/checkpoints -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`score`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/score) | Rank candidate continuations using `Session.ScoreTexts` and `Responses.Score` | `go run ./examples/score -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`token-scores`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/token-scores) | Extract per-token log-probabilities with `TokenScores` alongside `Engine.Tokenize` | `go run ./examples/token-scores -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`tokenize`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/tokenize) | Inspect token IDs, detokenization, and start/stop IDs via `Client.Engine()` | `go run ./examples/tokenize -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`fd-loader`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/fd-loader) | Sandboxed loading from open OS file descriptors via `WithModelFD` (Linux/macOS) | `go run ./examples/fd-loader -help` |
| [`lora-tuner`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/lora-tuner) | Configure base LoRA rank and supported rank lists on `EngineSettings` at startup | `go run ./examples/lora-tuner -help` |
| [`logging`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/logging) | Configure native runtime log severity floors via `SetMinLogLevel` | `go run ./examples/logging -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4` |
| [`debugger`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/debugger) | Capture and locate AI Edge Debugger traces with `WithDebuggerEnabled` and `DebugCaptureDir()` (`1.1+`) | `go run ./examples/debugger -help` |
