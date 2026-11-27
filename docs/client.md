# Client

The `Client` type is the high-level entry point to the LiteRT-LM
runtime. Always pair `New` with `defer client.Close()` to release
the underlying engine handles.

```go
client, err := litertlm.New(ctx,
    litertlm.WithLib("/abs/path/to/dist/lib"),
    litertlm.WithModel("/abs/path/to/model.litertlm"),
    litertlm.WithBackend("cpu"),
    litertlm.WithMaxTokens(4096),
)
defer client.Close()
```

## Initializing the Client

Use `litertlm.New` to initialize a new client. The client automatically manages the underlying inference engine and configuration settings for you. Always pair it with a deferred close to ensure resources are properly released when your application exits:

```go
client, err := litertlm.New(ctx, opts...)
defer client.Close()
```

## Construction options

Use functional options to specify environment and inference engine settings.

### Library and model

| Option                  | Effect                                                                |
|-------------------------|-----------------------------------------------------------------------|
| `WithLib(dir)`          | Directory holding `litert-lm.*` (or legacy `liblitertlm_c*.*`). Empty → `$LITERTLM_LIB`. |
| `WithLibName(name)`     | Override main shared library short-name (default `"litert-lm"`). Empty → `$LITERTLM_LIB_NAME`. |
| `WithModel(path)`       | Path to a `.litertlm` file. Empty → `$LITERTLM_MODEL`.                |
| `WithModelFD(fd)`       | Raw file descriptor of the model file (unsupported on Windows).       |
| `WithBackend(b)`        | `"cpu"` (default), `"gpu"` (`"apple"` / `"metal"` normalize to `"gpu"`), or `"ynnpack"` (sets CPU + enables YNNPACK delegation when `WithYNNPACK` is unset). |
| `WithVisionBackend(b)`  | Optional extra backend for vision inputs.                             |
| `WithAudioBackend(b)`   | Optional extra backend for audio inputs.                              |

### Engine settings

| Option                                    | Effect                                                                    |
|-------------------------------------------|---------------------------------------------------------------------------|
| `WithMaxTokens(n)`                        | Total token budget (prompt + output). Default 4096.                       |
| `WithCacheDir(dir)`                       | Engine artefact cache. Propagated to vision and audio executors.          |
| `WithActivationDataType(t)`               | 0=F32, 1=F16, 2=I16, 3=I8.                                                |
| `WithPrefillChunkSize(n)`                 | CPU-backend prefill chunk size for dynamic models.                        |
| `WithSpeculativeDecoding(cfg)`            | Configure `SpeculativeDecodingConfig{Mode, Lenience}` (`SpeculativeDecodingOff`, `SpeculativeDecodingOn`, `SpeculativeDecodingAdaptive`). Nonzero `Lenience` requires a runtime exporting `litert_lm_speculative_decoding_config_set_lenience`. |
| `WithSpeculativeDecodingEnabled(on)`      | Boolean toggle for multi-token speculative decoding (ignored when `WithSpeculativeDecoding` is set). See [`examples/speculative/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/speculative). |
| `WithBenchmarkEnabled()`                  | Turn on benchmark collection. Read per-call metrics via `Response.Benchmark()`. |
| `WithParallelSectionLoading(on)`          | Parallel deserialization of `.litertlm` container sections. Defaults to true. |
| `WithYNNPACK(on)`                         | Enable YNNPACK CPU delegation before XNNPACK.                             |
| `WithMaxVisionTokensPerImage(n)`          | Cap vision tokens generated per image.                                    |
| `WithMaxImages(n)`                        | Cap image inputs per call on legacy backends.                             |
| `WithMetalResidencySet(on)`               | Enable Metal residency sets on Apple GPU backends.                        |
| `WithGPUDecodeStepsPerSync(n)`            | Configure decode steps between GPU synchronizations.                      |
| `WithGPUWaitForWeightUploads(on)`         | Wait for weight uploads to complete during GPU initialization.            |
| `WithRingbuffersLocalAttention(on)`       | Store local attention KV caches in ringbuffers.                           |
| `WithSingleThreadedExecution(on)`         | Run session execution on caller-driven threads rather than background workers. |
| `WithDebuggerEnabled(on)`                 | Enable AI Edge Debugger trace captures under a writable `WithCacheDir` (requires C API `1.1+`; returns `ErrUnimplemented` on `1.0.0`). |
| `WithSelectedSignatures(signatures...)`   | Restrict the main executor to named model signatures (requires C API `1.1+`; returns `ErrUnimplemented` on `1.0.0`). |
| `WithDispatchLibDir(dir)`                 | LiteRT dispatch library directory for the NPU backend.                    |
| `WithVisionDispatchLibDir(dir)`           | LiteRT dispatch library directory for vision NPU embedding backends.      |
| `WithAudioDispatchLibDir(dir)`            | LiteRT dispatch library directory for audio NPU embedding backends.       |
| `WithMaxInputLength(n)`                   | Maximum text encoder sequence length for embedding engines.               |
| `WithMinInputLength(n)`                   | Minimum text encoder sequence length for embedding engines.               |
| `WithNumThreads(n)`                       | Set the number of threads for the CPU backend.                            |
| `WithAudioNumThreads(n)`                  | Set the number of threads for the audio CPU backend.                      |
| `WithLoRARank(rank)`                      | Set the startup LoRA rank for the engine.                                 |
| `WithSupportedLoRARanks(ranks)`           | Set the supported startup LoRA ranks for the engine.                      |
| `WithAudioLoRARank(rank)`                 | Set the startup audio LoRA rank for the engine.                           |
| `WithSupportedAudioLoRARanks(ranks)`      | Set the supported startup audio LoRA ranks for the engine.                |
| `WithConfigFile(path, profile)`           | Load engine and sampler settings from a centralized `config.json` profile file. |

### Logging

| Option                          | Effect                                                |
|---------------------------------|-------------------------------------------------------|
| `SetMinLogLevel(lvl)`           | Package-level function (not a Client option). `LogVerbose` / `LogDebug` / `LogInfo` / `LogWarning` / `LogError` / `LogFatal` / `LogQuiet`. Call before `New` to override the C-side default of `LogInfo`. |

### Sampler defaults

| Option                          | Effect                                                |
|---------------------------------|-------------------------------------------------------|
| `WithDefaultSampler(p)`         | Sampler used for every `Generate` unless overridden per-call by `WithSampler`. |

### Centralized configuration (`WithConfigFile`)

Store engine, speculative decoding, acceleration, thread-count, LoRA, and default sampler settings in a JSON file with a `"default"` block and per-model overrides under `"models"`. Load a profile with `litertlm.WithConfigFile("config.json", "model-profile")`; explicit Go options applied after `WithConfigFile` override profile fields.

Supported keys inside each model entry include:
`backend`, `max_tokens`, `cache_dir`, `speculative_decoding_enabled`, `speculative_decoding_mode` (`"off"` | `"on"` | `"adaptive"`), `speculative_decoding_lenience`, `parallel_section_loading`, `enable_ynnpack`, `max_vision_tokens_per_image`, `metal_residency_set`, `gpu_decode_steps_per_sync`, `gpu_wait_for_weight_uploads`, `use_ringbuffers_local_attention`, `single_threaded_execution`, `enable_debugger`, `selected_signatures`, `num_threads`, `audio_num_threads`, `vision_dispatch_lib_dir`, `audio_dispatch_lib_dir`, `max_input_length`, `min_input_length`, `lora_rank`, `supported_lora_ranks`, `audio_lora_rank`, `supported_audio_lora_ranks`, `sampler_type` (`"top_k"` | `"top_p"` | `"greedy"`), `temperature`, `top_k`, `top_p`, `seed`, `lib_path`, `lib_name`, and `model_path`.

See [`examples/config-file/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/config-file).

## Model capabilities, runtime version, and errors

Query a `.litertlm` file's capabilities without initializing an engine using `litertlm.InspectModel(path)`, or query the client's backing model with `client.Capabilities()` (when loaded by file path):

```go
caps, err := client.Capabilities()
if err != nil {
    return err
}
fmt.Printf("type=%s ctx=%d dynamic=%v modalities=%v thinking=%v tools=%v\n",
    caps.ModelType, caps.MaxContextTokens, caps.IsDynamicContext,
    caps.SupportedModalities, caps.SupportsThinking, caps.SupportsFunctionCalling)
```

`ModelCapabilities` reports `ModelType` (`ModelTypeLLM` / `ModelTypeEmbedding`), `MinRuntimeVersion`, `MaxContextTokens`, `IsDynamicContext`, `VisionSignatureLengths`, `EmbeddingDimension`, `EmbeddingSignatureLengths`, per-modality hardware support (`ModalityBackends`, `ModalityNPUBrand`, `ModalitySoCName`), and convenience predicates (`SupportsText()`, `SupportsVision()`, `SupportsAudio()`, `SupportsVideo()`). See [`examples/capabilities/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/capabilities).

Inspect the loaded C API version at runtime with `litertlm.APIVersion()`. Native runtime failures return a `*litertlm.CError` carrying the failing `Op`, canonical `StatusCode`, and thread-local C++ diagnostic `Message`. Match conditions portably with `errors.Is` against sentinel errors (`ErrCancelled`, `ErrInvalidArgument`, `ErrNotFound`, `ErrResourceExhausted`, `ErrFailedPrecondition`, `ErrUnimplemented`, `ErrInternal`, etc.) or extract details via `errors.As`.

### Runtime Residency & Debugger Diagnostics

Adjust Metal GPU weight residency at runtime with `client.SetMetalResidency(enable)` (active on Apple Metal GPU backends; safely ignored on other platforms or CPU execution). See [`examples/metal-residency/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/metal-residency).

When `WithDebuggerEnabled(true)` is paired with a writable `WithCacheDir` on C API `1.1+`, query relative capture directories using `chat.DebugCaptureDir()` or `session.DebugCaptureDir()`. Captures contain unencrypted prompt/decoded text and activation tensors; remove capture directories manually after diagnostics. See [`examples/debugger/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/debugger).

## Client methods
### `Generate(ctx, prompt, opts...)`

Synchronous one-shot inference. Returns the first candidate's text.

Per-call options are `RuntimeOption` values, shared with `GenerateData`
and `Chat.Send*`:

| Option                        | Effect                                                |
|-------------------------------|-------------------------------------------------------|
| `WithMaxOutputTokens(n)`      | Cap output tokens for this call.                      |
| `WithSampler(p)`              | Override the Client's default sampler.                |
| `WithTurnThinking(cfg)`       | Override thinking mode and token budget for the call. |
| `WithConstraint(typ, pattern)`| Apply a `ConstraintRegex` or `ConstraintJSONSchema` grammar. |
| `WithRepetitionPenalty(cfg)`  | Apply repetition, presence, and frequency penalties.  |
| `WithNoRepeatNgram(cfg)`      | Ban repeating token n-grams within a history window.  |
| `WithSuppressTokens(ids...)`  | Ban listed token IDs from being sampled.              |


```go
text, err := client.Generate(ctx, "The capital of France is")
```

`ctx` cancellation is propagated to `Conversation.Cancel` internally, so
`context.WithTimeout` and `context.WithCancel` apply:

```go
ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
defer cancel()

text, err := client.Generate(ctx, prompt)
if errors.Is(err, context.DeadlineExceeded) {
    // Model didn't finish in time.
}
```

### `GenerateStream(ctx, prompt, opts...)`

Provides access to token-by-token streaming.

```go
for chunk, err := range client.GenerateStream(ctx, prompt) {
    if err != nil {
        return err
    }
    fmt.Print(chunk.Text)
    if chunk.Final {
        fmt.Println()
    }
}
```

`Chunk` is a value type:

```go
type Chunk struct {
    Text  string
    Final bool
}
```

### `GenerateResponse(ctx, prompt, opts...)`

Use `GenerateResponse` when you need candidate metadata or per-call benchmark metrics alongside the text. It returns a `*Response` object:

```go
resp, err := client.GenerateResponse(ctx, prompt)
if err != nil { return err }

fmt.Println(resp.Text())          // Generated reply text
fmt.Println(resp.NumCandidates()) // Number of candidates present (1 when non-empty)
if b := resp.Benchmark(); b != nil && len(b.DecodeTokensPerSec) > 0 {
    fmt.Printf("kv_cap=%d decode=%.1f tok/s\n", b.MaxNumTokens, b.DecodeTokensPerSec[0])
}
```

For log-probability candidate scoring or per-token scores and lengths, use `Session.ScoreTexts` with low-level `Responses.Score` and `Responses.TokenLength` (see [Low-level API](low-level.md)).

## Multimodal inputs

When the model supports vision or audio (and `WithVisionBackend` /
`WithAudioBackend` are set), use the `*Multi` methods. They take a
`[]litertlm.Part` instead of a string prompt; everything else (opts,
streaming, cancellation, response shape) is identical.

| Method                       | Returns                       |
|------------------------------|-------------------------------|
| `GenerateMulti`              | `(string, error)`             |
| `GenerateMultiStream`        | `iter.Seq2[Chunk, error]`     |
| `GenerateMultiResponse`      | `(*Response, error)`          |

These are **one-shot**: each call opens a fresh `Conversation`,
runs one inference, and discards it. KV state does not persist
between calls. For successive multimodal turns that share
conversation state, use `Chat.SendMulti` / `Chat.SendMultiStream`
(see [Chat](chat.md)).

### Building Parts

| Constructor                          | Purpose                                                 |
|--------------------------------------|---------------------------------------------------------|
| `Text(s)`                            | Text prompt segment.                                    |
| `Image(b)`                           | Image bytes (no MIME claimed).                          |
| `ImageWithMIME(b, "image/jpeg")`     | Explicit MIME (jpeg / png / webp / gif / bmp).          |
| `ImageFromFile(path)`                | Read file; MIME from extension.                         |
| `Audio(b)` / `AudioWithMIME(b, mime)` / `AudioFromFile(path)` | Audio analogues. |

### Example — vision Q&A

```go
img, err := litertlm.ImageFromFile("/path/to/photo.jpg")
if err != nil { return err }

text, err := client.GenerateMulti(ctx, []litertlm.Part{
    img,
    litertlm.Text("What objects are visible?"),
})
```

## See also

- [Tutorial: Your First Local LLM App in 5 Minutes](tutorial-first-app.md) — walkthrough of `FetchLib`, `FetchModel`, streaming, and benchmarks.
- [Examples Gallery](examples.md) — all 48 runnable repository examples.
- [Chat](chat.md) — multi-turn with system prompts and tools.
- [Embeddings](embeddings.md) — standalone text, multimodal, and batch embeddings.
- [Structured output](structured-output.md) — `GenerateData[T]` and `GenerateDataMulti[T]`.
- [Low-level API](low-level.md) — when to drop down.
