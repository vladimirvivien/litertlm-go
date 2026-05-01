# Client

The `Client` is the front door to the high-level API. Construct one
per process per model, reuse it for every generation call, `Close()`
on shutdown.

```go
client, err := litertlm.New(ctx,
    litertlm.WithLib("/abs/path/to/dist/lib"),
    litertlm.WithModel("/abs/path/to/model.litertlm"),
    litertlm.WithBackend("cpu"),
    litertlm.WithMaxTokens(4096),
)
defer client.Close()
```

## `New(ctx, opts...)`

Aggregates the load + `EngineSettings` + `Engine` flow into one call.
The `ctx` is reserved for future cancellation of slow loads — the
current C API runs synchronous load, so the context is not consulted
yet, but plumbing it through keeps callers idiomatic.

`Close()` is idempotent. Call it from `defer` after a successful
`New`.

## Construction options

Functional options absorb every existing `EngineSettings` setter plus
load-time choices.

### Library and model

| Option                  | Effect                                                                |
|-------------------------|-----------------------------------------------------------------------|
| `WithLib(dir)`          | Directory holding `liblitertlm_c_cpu.*` etc. Empty → `$LITERTLM_LIB`. |
| `WithModel(path)`       | Path to a `.litertlm` file. Empty → `$LITERTLM_MODEL`.                |
| `WithBackend(b)`        | `"cpu"` (default) or `"gpu"`.                                         |
| `WithVisionBackend(b)`  | Optional extra backend for vision inputs.                             |
| `WithAudioBackend(b)`   | Optional extra backend for audio inputs.                              |

### Engine settings

| Option                                    | Effect                                                                    |
|-------------------------------------------|---------------------------------------------------------------------------|
| `WithMaxTokens(n)`                        | Total token budget (prompt + output). Default 4096.                       |
| `WithCacheDir(dir)`                       | Engine artefact cache.                                                    |
| `WithActivationDataType(t)`               | 0=F32, 1=F16, 2=I16, 3=I8.                                                |
| `WithPrefillChunkSize(n)`                 | CPU-backend prefill chunk size for dynamic models.                        |
| `WithEnableSpeculativeDecoding(on)`       | Toggle speculative decoding.                                              |
| `WithEnableBenchmark()`                   | Turn on benchmark collection (read via the low-level `BenchmarkInfo`).    |
| `WithParallelFileLoading(on)`             | Override the C-side default (true).                                       |

### Logging

| Option                          | Effect                                                |
|---------------------------------|-------------------------------------------------------|
| `WithLogLevel(lvl)`             | `LogVerbose` / `LogDebug` / `LogInfo` / `LogWarning` / `LogError` (default) / `LogFatal` / `LogSilent`. |

### Sampler defaults

| Option                          | Effect                                                |
|---------------------------------|-------------------------------------------------------|
| `WithDefaultSampler(p)`         | Sampler used for every `Generate` unless overridden per-call by `WithSampler`. |

## `Generate(ctx, prompt, opts...)`

Synchronous one-shot inference. Returns the first candidate's text.

```go
text, err := client.Generate(ctx, "The capital of France is")
```

`ctx` cancellation is wired through to `Session.Cancel` internally,
so `context.WithTimeout` and `context.WithCancel` Just Work:

```go
ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
defer cancel()

text, err := client.Generate(ctx, prompt)
if errors.Is(err, context.DeadlineExceeded) {
    // Model didn't finish in time.
}
```

Per-call options:

| Option                          | Effect                                                |
|---------------------------------|-------------------------------------------------------|
| `WithMaxOutputTokens(n)`        | Cap output tokens for this call.                      |
| `WithSampler(p)`                | Override the Client's default sampler.                |

## `GenerateStream(ctx, prompt, opts...)`

Token-by-token streaming via Go 1.23+ range-over-func.

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

The iterator yields chunks with `Final=false` followed by exactly one
`Final=true` chunk. Errors arrive on the second return; on error the
loop should exit (subsequent iterations would also surface the error
or close immediately).

Cancelling `ctx` aborts mid-stream; the iterator yields the
`CANCELLED` error, then the channel closes. See
[`examples/cancel/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/cancel)
for a full demo.

## `GenerateResponse(ctx, prompt, opts...)`

Rich-output sibling of `Generate`. Returns a `*Response` that exposes
per-candidate text plus score and token-length accessors:

```go
resp, err := client.GenerateResponse(ctx, prompt)
if err != nil { return err }

fmt.Println(resp.Text())                    // first candidate
fmt.Println(resp.NumCandidates())           // usually 1 with default sampler
score, ok := resp.Score(0)                  // (placeholder, always ok=true for non-scoring sources)
length, ok := resp.TokenLength(0)           // (false unless ScoreTexts populated it)
```

`*Response` owns the underlying C handle. When the variable becomes
unreachable, a `runtime.AddCleanup`-registered callback frees the
handle. Callers don't (and shouldn't) call `Delete` on it.

For deterministic release — generating thousands of responses in a
tight memory-bound loop — drop down to the low-level
`Session.GenerateContent` and call `.Delete()` yourself.
See [Low-level API](low-level.md).

## Concurrency

`Client` is safe for concurrent use. Each generation call opens a
fresh `Session` under an internal mutex (sessions are millisecond-
scale to construct, far cheaper than inference itself). The C engine
restricts session reuse across prefill/decode cycles, so the per-call
session model is forced rather than chosen.

If multiple goroutines call `Generate` simultaneously, they will
serialize on session creation only — the actual inference runs in
parallel as far as the C engine allows.

## See also

- [Chat](chat.md) — multi-turn with system prompts and tools.
- [Structured output](structured-output.md) — `GenerateData[T]`.
- [Low-level API](low-level.md) — when to drop down.
