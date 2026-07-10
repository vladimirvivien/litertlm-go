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
| `WithLib(dir)`          | Directory holding `liblitertlm_c_cpu.*` etc. Empty → `$LITERTLM_LIB`. |
| `WithModel(path)`       | Path to a `.litertlm` file. Empty → `$LITERTLM_MODEL`.                |
| `WithModelFd(fd)`       | Raw file descriptor of the model file (unsupported on Windows).       |
| `WithBackend(b)`        | `"cpu"` (default) or `"gpu"`.                                         |
| `WithVisionBackend(b)`  | Optional extra backend for vision inputs.                             |
| `WithAudioBackend(b)`   | Optional extra backend for audio inputs.                              |

### Engine settings

| Option                                    | Effect                                                                    |
|-------------------------------------------|---------------------------------------------------------------------------|
| `WithMaxTokens(n)`                        | Total token budget (prompt + output). Default 4096.                       |
| `WithCacheDir(dir)`                       | Engine artefact cache. Propagated to vision and audio executors.          |
| `WithActivationDataType(t)`               | 0=F32, 1=F16, 2=I16, 3=I8.                                                |
| `WithPrefillChunkSize(n)`                 | CPU-backend prefill chunk size for dynamic models.                        |
| `WithSpeculativeDecodingEnabled(on)`      | Toggle multi-token-prediction speculative decoding (Gemma 4 supported). See [`examples/speculative/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/speculative) for a side-by-side throughput comparison. |
| `WithBenchmarkEnabled()`                  | Turn on benchmark collection. Read per-call metrics via `Response.Benchmark()`. |
| `WithParallelSectionLoading(on)`          | Parallel deserialization of `.litertlm` container sections. Defaults to true. |
| `WithDispatchLibDir(dir)`                 | LiteRT dispatch library directory for the NPU backend.                    |
| `WithNumThreads(n)`                       | Set the number of threads for the CPU backend.                            |
| `WithAudioNumThreads(n)`                  | Set the number of threads for the audio CPU backend.                      |
| `WithLoRARank(rank)`                      | Set the startup LoRA rank for the engine.                                 |
| `WithSupportedLoRARanks(ranks)`           | Set the supported startup LoRA ranks for the engine.                      |
| `WithAudioLoRARank(rank)`                 | Set the startup audio LoRA rank for the engine.                           |
| `WithSupportedAudioLoRARanks(ranks)`      | Set the supported startup audio LoRA ranks for the engine.                |

### Logging

| Option                          | Effect                                                |
|---------------------------------|-------------------------------------------------------|
| `SetMinLogLevel(lvl)`           | Package-level function (not a Client option). `LogVerbose` / `LogDebug` / `LogInfo` / `LogWarning` / `LogError` / `LogFatal` / `LogQuiet`. Call before `New` to override the C-side default of `LogInfo`. |

### Sampler defaults

| Option                          | Effect                                                |
|---------------------------------|-------------------------------------------------------|
| `WithDefaultSampler(p)`         | Sampler used for every `Generate` unless overridden per-call by `WithSampler`. |

## Client methods
### `Generate(ctx, prompt, opts...)`

Synchronous one-shot inference. Returns the first candidate's text.

Per-call options are `RuntimeOption` values, shared with `GenerateData`
and `Chat.Send*`:

| Option                          | Effect                                                |
|---------------------------------|-------------------------------------------------------|
| `WithMaxOutputTokens(n)`        | Cap output tokens for this call.                      |
| `WithSampler(p)`                | Override the Client's default sampler.                |


```go
text, err := client.Generate(ctx, "The capital of France is")
```

`ctx` cancellation is propagated to `Session.Cancel` internally, so
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

Use `GenerateResponse` when you need detailed metadata about the generation, such as scoring or token length, rather than just the raw text. It returns a rich `*Response` object:

```go
resp, err := client.GenerateResponse(ctx, prompt)
if err != nil { return err }

fmt.Println(resp.Text())          // Get the generated reply text
fmt.Println(resp.NumCandidates()) // Number of candidates emitted
score, ok := resp.Score(0)        // Score of the first candidate
```

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
| `ImageWithMime(b, "image/jpeg")`     | Explicit MIME (jpeg / png / webp / gif / bmp).          |
| `ImageFromFile(path)`                | Read file; MIME from extension.                         |
| `Audio(b)` / `AudioWithMime(b, mime)` / `AudioFromFile(path)` | Audio analogues. |

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

- [Chat](chat.md) — multi-turn with system prompts and tools.
- [Structured output](structured-output.md) — `GenerateData[T]` and `GenerateDataMulti[T]`.
- [Low-level API](low-level.md) — when to drop down.
