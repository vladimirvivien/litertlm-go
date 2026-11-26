# Standalone Multimodal Embeddings

This example generates dense vector embeddings from text (and optional image inputs) using `litertlm.NewEmbeddingClient`, compares cosine similarity across related and unrelated inputs, and demonstrates `WithOutputSize` (Matryoshka truncation), `WithNormalize`, and `WithOverflowStrategy`.

## Prerequisites

* LiteRT-LM unified C-API shared library (`v0.19.1+`).
* An embedding-type `.litertlm` model (`ModelTypeEmbedding`).

## Running

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
LITERTLM_MODEL=/abs/path/to/embedding-model.litertlm \
go run ./examples/embed
```

Override Matryoshka output size, normalization, or overflow strategy:

```bash
go run ./examples/embed -dim 256 -normalize=true -overflow chunk
```
