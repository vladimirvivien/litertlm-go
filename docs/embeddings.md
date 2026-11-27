# Embeddings

`EmbeddingClient` provides a high-level API for generating dense vector embeddings from text and multimodal inputs using standalone `.litertlm` embedding models (`ModelTypeEmbedding`). Use `litertlm.NewEmbeddingClient` when your workflow needs semantic search, clustering, retrieval, or multimodal similarity without initializing a generative chat engine.

```go
client, err := litertlm.NewEmbeddingClient(ctx,
    litertlm.WithLib("/abs/path/to/dist/lib"),
    litertlm.WithModel("/abs/path/to/embedding-model.litertlm"),
    litertlm.WithBackend("cpu"),
)
if err != nil {
    return err
}
defer client.Close()

vec, err := client.Embed(ctx, "Go concurrency with goroutines and channels",
    litertlm.WithNormalize(true),
    litertlm.WithOverflowStrategy(litertlm.OverflowChunkAndAverage),
)
```

## Initializing `NewEmbeddingClient(ctx, opts...)`

`NewEmbeddingClient` shares the same [`Option`](client.md#construction-options) constructors as `litertlm.New`. Environment fallbacks `$LITERTLM_LIB`, `$LITERTLM_LIB_NAME`, and `$LITERTLM_MODEL`, along with `WithConfigFile`, apply in the same order. File-descriptor loading (`WithModelFD`) is not supported for embedding engines.

Embedding-specific construction options:

| Option | Effect |
|---|---|
| `WithNumThreads(n)` | Set CPU threads for text embedding execution. |
| `WithAudioNumThreads(n)` | Set CPU threads for audio embedding execution. |
| `WithCacheDir(dir)` | Share a compiled model cache across main, vision, and audio executors. |
| `WithVisionDispatchLibDir(dir)` | Specify the LiteRT dispatch library directory for vision NPU backends. |
| `WithAudioDispatchLibDir(dir)` | Specify the LiteRT dispatch library directory for audio NPU backends. |
| `WithMaxInputLength(n)` | Cap text encoder signature length (non-positive unsets). |
| `WithMinInputLength(n)` | Set minimum text encoder signature length (negative unsets). |
| `WithMaxVisionTokensPerImage(n)` | Set target vision tokens per image on embedding engines. |
| `WithActivationDataType(t)` | Select activation precision (`0=F32`, `1=F16`, `2=I16`, `3=I8`). |

Verify model type and native output dimensions before loading with `litertlm.InspectModel` or `client.Capabilities()`:

```go
caps, err := litertlm.InspectModel(modelPath)
if err != nil {
    return err
}
if caps.ModelType != litertlm.ModelTypeEmbedding {
    return fmt.Errorf("expected embedding model, got %s", caps.ModelType)
}
fmt.Printf("dim=%d signatures=%v\n", caps.EmbeddingDimension, caps.EmbeddingSignatureLengths)
```

## Generating Embeddings

Use `Embed` for single strings, `EmbedBatch` to vectorize multiple strings in one native batch call, or `EmbedMulti` / `EmbedMultiBatch` when inputs include `Image` or `Audio` [`Part`](client.md#building-parts) segments alongside text.

```go
vecs, err := client.EmbedBatch(ctx, []string{
    "Go concurrency with goroutines and channels",
    "Lightweight concurrent execution in Go",
},
    litertlm.WithNormalize(true),
    litertlm.WithOverflowStrategy(litertlm.OverflowChunkAndAverage),
)
if err != nil {
    return err
}
fmt.Printf("embedded %d vectors of length %d\n", len(vecs), len(vecs[0]))
```

Multimodal example combining image bytes and text:

```go
img, err := litertlm.ImageFromFile("/path/to/photo.jpg")
if err != nil {
    return err
}

vec, err := client.EmbedMulti(ctx, []litertlm.Part{
    img,
    litertlm.Text("Sunset over the Pacific coastline"),
},
    litertlm.WithEmbeddingVisionTokensPerImage(280),
)
```

## Per-Call Options (`EmbeddingOption`)

Every `Embed*` method accepts variadic `EmbeddingOption` functions controlling normalization, special-token framing, long-input overflow handling, and Matryoshka truncation. When no options are passed, the runtime uses C defaults (`normalize=true`, `insert_special_tokens=true`, `overflow=OverflowError`).

| Option | Effect |
|---|---|
| `WithNormalize(on)` | Toggle L2 normalization on returned embedding vectors. |
| `WithInsertSpecialTokens(on)` | Toggle automatic BOS/EOS and modality marker insertion. |
| `WithOverflowStrategy(strategy)` | Select handling when input exceeds maximum signature length (`OverflowChunkAndAverage`, `OverflowTruncate`, `OverflowError`). |
| `WithOutputSize(n)` | Truncate vectors to the first `n` Matryoshka dimensions (`n <= 0` keeps full model dimension). |
| `WithEmbeddingVisionTokensPerImage(n)` | Override vision tokens generated per image for a single request (`n <= 0` unsets). |

## Low-Level Control

Drop down to `EmbeddingEngineSettings`, `EmbeddingEngine`, `EmbeddingOptions`, `EmbeddingResponse`, and `EmbeddingResponses` when you manage reusable `EmbeddingOptions` handles or `InputData` allocations directly across repeated calls. Access the underlying handles via `client.Engine()` and `client.Settings()`; see the [Low-level API](low-level.md#method-map) for handle lifetime rules.

## See also

- [Tutorial: Local Multimodal Semantic Search & Zero-Shot Classification](tutorial-semantic-search.md) — build a cross-modal index and zero-shot classifier step-by-step.
- [Examples Gallery](examples.md) — all 48 runnable repository examples.
- [`examples/embed/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/embed) — text, batch, Matryoshka truncation, and optional multimodal embedding similarity.
- [`examples/semantic-search/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/semantic-search) — reusable `search` index and zero-shot `classify` subpackages with a multimodal CLI.
- [`examples/capabilities/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/capabilities) — inspect embedding dimensions and signature lengths before loading.
- [Client Guide](client.md) — shared initialization options and error inspection.
