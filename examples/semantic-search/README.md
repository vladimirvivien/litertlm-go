# Multimodal Semantic Search & Zero-Shot Classification

This example builds two practical workflows on top of `litertlm.EmbeddingClient`:

1. **[`search`](./search/):** Index folders of mixed text, image, and audio files into a shared embedding space, persist indexes as JSON, and run text or cross-modal media queries (one-shot or interactive REPL).
2. **[`classify`](./classify/):** Classify text snippets, images, or audio files zero-shot against candidate labels using cached label embeddings and temperature-scaled softmax probabilities.

Both workflows share a single embedding vector space provided by multimodal `.litertlm` embedding models such as EmbeddingGemma 2.

## Prerequisites

* LiteRT-LM unified C-API shared library (`v0.19.1+`).
* An embedding `.litertlm` model (`ModelTypeEmbedding`). Public, ungated EmbeddingGemma 2 models on Hugging Face include:
  * **Text only (165 MB):** `litert-community/embeddinggemma-2-text-270m-litert-lm`
  * **Text + Vision (388 MB):** `litert-community/embeddinggemma-2-text-vision-440m-litert-lm`
  * **Text + Vision + Audio (485 MB):** `litert-community/embeddinggemma-2-740m-litert-lm`

Inspect model capabilities automatically enabled by this tool (`SupportsVision`, `SupportsAudio`, `EmbeddingDimension`):

```bash
go run ./examples/capabilities -get-lib v0.19.1 -get-model litert-community/embeddinggemma-2-text-vision-440m-litert-lm
```

## Semantic Search (`search` subcommand)

Index a directory of text descriptions and images, then run a text query:

```bash
go run ./examples/semantic-search search \
  -get-lib v0.19.1 \
  -get-model litert-community/embeddinggemma-2-text-vision-440m-litert-lm \
  -dir ./examples/testdata \
  -query "overhead aerial photograph of a busy parking lot with rows of cars"
```

Run a cross-modal image-to-text/image query against the same directory:

```bash
go run ./examples/semantic-search search \
  -get-lib v0.19.1 \
  -get-model litert-community/embeddinggemma-2-text-vision-440m-litert-lm \
  -dir ./examples/testdata \
  -query-image ./examples/testdata/img1.png
```

Save an index once and reload it in an interactive query REPL (`<text>`, `:image <path>`, `:audio <path>`, `:stats`, `:quit`):

```bash
go run ./examples/semantic-search search \
  -get-lib v0.19.1 \
  -get-model litert-community/embeddinggemma-2-text-vision-440m-litert-lm \
  -dir ./examples/testdata \
  -index /tmp/testdata-index.json

go run ./examples/semantic-search search \
  -get-lib v0.19.1 \
  -get-model litert-community/embeddinggemma-2-text-vision-440m-litert-lm \
  -index /tmp/testdata-index.json
```

Use `-dim 256` (or `128`, `512`) to test Matryoshka dimension truncation.

## Zero-Shot Classification (`classify` subcommand)

Classify images and text inputs against natural-language candidate labels:

```bash
go run ./examples/semantic-search classify \
  -get-lib v0.19.1 \
  -get-model litert-community/embeddinggemma-2-text-vision-440m-litert-lm \
  -labels "indoor corner table with lamp and glasses,aerial parking lot with cars,ocean beach sunset" \
  -text "many cars parked in rows seen from above" \
  ./examples/testdata/img1.png \
  ./examples/testdata/img2.png
```

Use repeatable `-label` flags when labels contain commas, and `-temperature` to adjust softmax sharpness.

## Sample Output

*(Host-specific timings differ.)*

```text
Indexed 4 item(s) (2 text, 2 image, 0 audio; skipped 0) in 11.397s [dim=768]

Results for text query "overhead aerial view of parked cars" (629ms):
  1. [############..]  0.8776  [text ]  img2.md  -- The photograph depicts an aerial shot, taken directly overhead, of a busy urban parking lot with three horizontal row...
  2. [###########...]  0.7954  [image]  img2.png
  3. [########......]  0.6065  [text ]  img1.md  -- The photobrah depicts a small corner table in a room with soft light green walls. The table has a lamp, a pair of rea...
  4. [########......]  0.5545  [image]  img1.png
```

Zero-shot classification output on images:

```text
Prepared 3 label(s) [dim=768, temp=0.070] in 204ms

Input: ./examples/testdata/img1.png -> Top: indoor corner table with lamp and glasses (sim=0.7716, prob=93.0%) [1.045s]
  1. [#############.]   93.0%  sim=+0.7716  indoor corner table with lamp and glasses
  2. [#.............]    5.2%  sim=+0.5696  cooking recipe
  3. [..............]    1.8%  sim=+0.4951  aerial parking lot with cars
```

## Running Tests

Run unit tests for the CLI and both subpackages without a native library or model:

```bash
go test -short -race ./examples/semantic-search/...
```

Run live integration tests with an embedding model:

```bash
LITERTLM_TEST_LIB=/abs/path/to/dist/lib \
LITERTLM_TEST_EMBED_MODEL=/abs/path/to/embeddinggemma-2-text-vision-440m.litertlm \
go test -v ./examples/semantic-search/...
```
