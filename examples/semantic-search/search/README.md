# `search` Subpackage

`github.com/vladimirvivien/litertlm-go/examples/semantic-search/search` implements an in-process multimodal semantic index backed by `litertlm.EmbeddingClient`.

Because multimodal embedding models map text, images, and audio into a single shared vector space, `Index` stores vectors from every modality together and ranks cross-modal queries using cosine similarity.

## Features

- **Unified Multimodal Indexing:** `IndexDir` walks a directory, batches text files (`.txt`, `.md`, `.go`, `.json`, `.yaml`, `.csv`, `.toml`) through `EmbedBatch`, and embeds enabled image (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`) and audio (`.wav`, `.mp3`, `.flac`, `.ogg`) files via `EmbedMulti`.
- **Cross-Modal Queries:** Query any index using `QueryText`, `QueryImage`, `QueryAudio`, or arbitrary `QueryParts`.
- **Retrieval Task Prefixes:** `DefaultConfig` sets EmbeddingGemma retrieval prefixes (`DefaultQueryPrefix` and `DefaultDocumentPrefix`) for stronger query-to-document separation.
- **Persistent JSON Schema:** `Save` and `Load` serialize versioned JSON (`version`, `model`, `dim`, `entries`) so cached indexes reload without re-embedding unmodified files.
- **Matryoshka & Overflow Controls:** Configure `OutputDim` (`WithOutputSize`), `Overflow` (`WithOverflowStrategy`), `Normalize`, and `VisionTokensPerImage` once in `Config`.

## Usage

```go
cfg := search.DefaultConfig()
cfg.Model = "embeddinggemma-2-text-vision-440m.litertlm"
cfg.EnableVision = caps.SupportsVision()
cfg.EnableAudio = caps.SupportsAudio()

idx := search.New(embedClient, cfg)
stats, err := idx.IndexDir(ctx, "./docs")
if err != nil {
	log.Fatal(err)
}
log.Printf("indexed %d entries (dim=%d)", stats.Indexed, idx.Dim())

if err := idx.Save("index.json"); err != nil {
	log.Fatal(err)
}

matches, err := idx.QueryText(ctx, "overhead parking lot with cars", 5)
if err != nil {
	log.Fatal(err)
}
for _, m := range matches {
	fmt.Printf("%.4f  [%s]  %s\n", m.Score, m.Entry.Modality, m.Entry.ID)
}
```

## Testing

Run deterministic unit tests:

```bash
go test -v ./examples/semantic-search/search
```

Run live integration tests against a real embedding `.litertlm` model:

```bash
LITERTLM_TEST_LIB=/abs/path/to/dist/lib \
LITERTLM_TEST_EMBED_MODEL=/abs/path/to/embeddinggemma-2-text-vision-440m.litertlm \
go test -v ./examples/semantic-search/search -run TestLive_Search_EndToEnd
```
