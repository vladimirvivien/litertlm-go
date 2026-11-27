# Local Multimodal Semantic Search & Zero-Shot Classification

Dedicated embedding models convert text paragraphs, images, and audio recordings into dense numerical vectors inside a single shared coordinate space. Once artifacts live in that space, you can search across modalities by meaning rather than exact keywords—matching a text query like *"aerial view of parked cars"* directly to overhead photography—and classify unseen files zero-shot with no task-specific training data. In this tutorial, you vectorize inputs with `litertlm.EmbeddingClient`, build a persistent cross-modal index using the `search` and `classify` packages under `examples/semantic-search`, and shrink vector storage using Matryoshka truncation.

<div class="lrt-meta-bar">
  <span class="lrt-pill lrt-pill--coral">~10 min</span>
  <span class="lrt-pill lrt-pill--teal">Embeddings &amp; Retrieval</span>
  <span class="lrt-pill lrt-pill--gold">EmbeddingGemma 2 (Ungated)</span>
  <span class="lrt-pill lrt-pill--plum">Text · Image · Audio</span>
</div>

---

## Choosing a Multimodal Embedding Model

LiteRT-LM models built for vector retrieval report `ModelTypeEmbedding` from `InspectModel`. Public, ungated **EmbeddingGemma 2** weights on Hugging Face download directly via `litertlm.FetchModel` without access tokens:

| Model Identifier | Modalities | Download Size | Native Vector Dim |
|---|---|:---:|:---:|
| `litert-community/embeddinggemma-2-text-270m-litert-lm` | Text | ~165 MB | 768 |
| `litert-community/embeddinggemma-2-text-vision-440m-litert-lm` | Text + Vision | ~388 MB | 768 |
| `litert-community/embeddinggemma-2-740m-litert-lm` | Text + Vision + Audio | ~485 MB | 768 |

Check any `.litertlm` artifact with `examples/capabilities` before indexing:

```bash
go run ./examples/capabilities -get-lib v0.19.1 -get-model litert-community/embeddinggemma-2-text-vision-440m-litert-lm
```

---

## Step 1 — Compute Vectors Directly with `EmbeddingClient`

Unlike generative LLMs (`litertlm.New`), standalone embedding models open via `litertlm.NewEmbeddingClient`. Pass `WithNormalize(true)` so cosine similarity simplifies to a fast dot product, and `WithOverflowStrategy(litertlm.OverflowChunkAndAverage)` so long documents exceeding the encoder sequence window are automatically chunked and averaged instead of failing:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"runtime"

	"github.com/vladimirvivien/litertlm-go/examples/semantic-search/classify"
	"github.com/vladimirvivien/litertlm-go/examples/semantic-search/search"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	ctx := context.Background()

	libDir, err := litertlm.FetchLib(ctx, runtime.GOOS, runtime.GOARCH, "v0.19.1")
	if err != nil { log.Fatal(err) }

	modelPath, err := litertlm.FetchModel(ctx, "litert-community/embeddinggemma-2-text-vision-440m-litert-lm")
	if err != nil { log.Fatal(err) }

	if err := litertlm.Load(libDir, "cpu", ""); err != nil { log.Fatal(err) }
	caps, err := litertlm.InspectModel(modelPath)
	if err != nil { log.Fatal(err) }
	if caps.ModelType != litertlm.ModelTypeEmbedding {
		log.Fatalf("expected embedding model, got %s", caps.ModelType)
	}

	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	embedder, err := litertlm.NewEmbeddingClient(ctx,
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
		litertlm.WithBackend("cpu"),
	)
	if err != nil { log.Fatal(err) }
	defer embedder.Close()

	// Single normalized embedding vector:
	vec, err := embedder.Embed(ctx, "Go concurrency with goroutines",
		litertlm.WithNormalize(true),
		litertlm.WithOverflowStrategy(litertlm.OverflowChunkAndAverage),
	)
	if err != nil { log.Fatal(err) }
	fmt.Printf("Embedded vector length: %d (vision supported=%v)\n", len(vec), caps.SupportsVision())
```

---

## Step 2 — Index Mixed Files & Query Across Modalities (`search`)

Rather than writing boilerplate directory walkers and cosine-ranking loops by hand, import [`examples/semantic-search/search`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/semantic-search/search). Its `Index` applies EmbeddingGemma's asymmetric retrieval prefixes (`DefaultQueryPrefix` / `DefaultDocumentPrefix`), batch-embeds text documents, calls `EmbedMulti` on image or audio files, saves JSON snapshots cleanly, and scores queries by cosine similarity:

```go
	cfg := search.DefaultConfig()
	cfg.EnableVision = caps.SupportsVision()
	cfg.EnableAudio = caps.SupportsAudio()

	idx := search.New(embedder, cfg)
	stats, err := idx.IndexDir(ctx, "./examples/testdata")
	if err != nil { log.Fatal(err) }
	fmt.Printf("Indexed %d items (%d text, %d image) in %v\n",
		stats.Indexed, stats.TextCount, stats.ImageCount, stats.Duration)

	// Query both text descriptions and raw images with a natural-language phrase:
	matches, err := idx.QueryText(ctx, "overhead aerial photograph of a parking lot with rows of cars", 4)
	if err != nil { log.Fatal(err) }
	for i, m := range matches {
		fmt.Printf("%d. %.4f [%-5s] %s\n", i+1, m.Score, m.Entry.Modality, m.Entry.ID)
	}

	// Persist to disk for instant offline reloads:
	if err := idx.Save("index.json"); err != nil { log.Fatal(err) }
```

### Run & Sample Search Output

Run the CLI directly from the repository root against `examples/testdata`:

```bash
go run ./examples/semantic-search search \
  -get-lib v0.19.1 \
  -get-model litert-community/embeddinggemma-2-text-vision-440m-litert-lm \
  -dir ./examples/testdata \
  -query "overhead aerial photograph of a busy parking lot with rows of cars"
```

Swap `-query` for `-query-image ./examples/testdata/img1.png` to find text captions and visually similar images from a picture query, or save and reopen an interactive REPL (`-index /tmp/index.json`).

*(Host-specific timings differ.)*

```text
Indexed 4 item(s) (2 text, 2 image, 0 audio; skipped 0) in 11.397s [dim=768]

Results for text query "overhead aerial view of parked cars" (629ms):
  1. [############..]  0.8776  [text ]  img2.md  -- The photograph depicts an aerial shot, taken directly overhead, of a busy urban parking lot...
  2. [###########...]  0.7954  [image]  img2.png
  3. [########......]  0.6065  [text ]  img1.md  -- The photobrah depicts a small corner table in a room with soft light green walls...
  4. [########......]  0.5545  [image]  img1.png
```

---

## Step 3 — Zero-Shot Multimodal Classification (`classify`)

[`examples/semantic-search/classify`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/semantic-search/classify) pre-embeds candidate label phrases once via `EmbedBatch` and ranks incoming text, images, or audio against them using temperature-scaled softmax probabilities (`DefaultTemperature = 0.07`):

```go
	cCfg := classify.DefaultConfig()
	clf, err := classify.NewFromNames(ctx, embedder, []string{
		"indoor corner table with lamp and glasses",
		"aerial parking lot with cars",
		"ocean beach sunset",
	}, cCfg)
	if err != nil { log.Fatal(err) }

	res, err := clf.ClassifyImage(ctx, "./examples/testdata/img1.png", 3)
	if err != nil { log.Fatal(err) }
	fmt.Printf("Top prediction: %s (%.1f%% confidence)\n",
		res.Top.Label, res.Top.Probability*100)
}
```

Run the CLI `classify` subcommand across multiple images and text strings in one call:

```bash
go run ./examples/semantic-search classify \
  -get-lib v0.19.1 \
  -get-model litert-community/embeddinggemma-2-text-vision-440m-litert-lm \
  -labels "indoor corner table with lamp and glasses,aerial parking lot with cars,ocean beach sunset" \
  -text "many cars parked in rows seen from above" \
  ./examples/testdata/img1.png \
  ./examples/testdata/img2.png
```

*(Host-specific timings differ.)*

```text
Prepared 3 label(s) [dim=768, temp=0.070] in 204ms

Input: ./examples/testdata/img1.png -> Top: indoor corner table with lamp and glasses (sim=0.7716, prob=93.0%) [1.045s]
  1. [#############.]   93.0%  sim=+0.7716  indoor corner table with lamp and glasses
  2. [#.............]    5.2%  sim=+0.5696  ocean beach sunset
  3. [..............]    1.8%  sim=+0.4951  aerial parking lot with cars
```

---

## Step 4 — Shrink Indexes with Matryoshka Truncation

EmbeddingGemma 2 models are trained with Matryoshka representation learning, meaning the leading coordinates carry the strongest semantic signal. Passing `litertlm.WithOutputSize(256)` (or `-dim 256` on the CLI, supporting `128`, `256`, or `512`) cuts index size and scoring cost by 3x while retaining strong retrieval accuracy:

```go
smallVec, err := embedder.Embed(ctx, "parked sedans from above",
	litertlm.WithNormalize(true),
	litertlm.WithOutputSize(256),
)
```

On `search.Config` and `classify.Config`, set `OutputDim: 256` so both indexed documents and incoming queries truncate to matching 256-dimensional vectors.

---

## Try the Repository Examples

* [`examples/embed/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/embed) — text, batch, optional `-image`, overflow strategies, and `-dim` truncation side-by-side.
* [`examples/semantic-search/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/semantic-search) — reusable `search` and `classify` Go packages, offline unit tests (`go test -short -race ./examples/semantic-search/...`), and multimodal CLI.

## Next Steps

* Consult [Embeddings Guide](embeddings.md) for every `EmbeddingOption` and construction knob (`WithMaxInputLength`, `WithEmbeddingVisionTokensPerImage`, NPU dispatch dirs).
* Continue to [Tool-Calling Agents & KV-Cache Branching](tutorial-tools-branching.md) to connect `litertlm-go` models to live Go functions.
