# `classify` Subpackage

`github.com/vladimirvivien/litertlm-go/examples/semantic-search/classify` implements zero-shot classification for text, images, and audio using `litertlm.EmbeddingClient`.

Candidate labels are embedded once during `New` (or `NewFromNames`) via `EmbedBatch`. Incoming inputs are embedded into the same space, scored by cosine similarity against every cached label vector, and normalized into probabilities with temperature-scaled `Softmax`.

## How Zero-Shot Scoring Works

1. **Pre-embed Labels Once:** Each `Label` embeds its `Description` (falling back to `Name`), with optional `LabelPrefix`. Labels do not need to be re-embedded across inputs.
2. **Cosine Similarity:** Each input embedding is compared against all cached label vectors using `search.CosineSimilarity`.
3. **Temperature-Scaled Softmax:** Similarities are divided by `Config.Temperature` (default `0.07`) before applying `Softmax`, turning small cosine differences into calibrated probability weights across the full label set even when truncating `Scores` via `topK`.

## Usage

```go
labels := []classify.Label{
	{Name: "indoor", Description: "indoor corner table with lamp and glasses"},
	{Name: "parking", Description: "aerial photograph of a busy urban parking lot"},
	{Name: "beach", Description: "sunny coastline with ocean waves"},
}

clf, err := classify.New(ctx, embedClient, labels, classify.DefaultConfig())
if err != nil {
	log.Fatal(err)
}

res, err := clf.ClassifyFile(ctx, "./examples/testdata/img2.png", 0)
if err != nil {
	log.Fatal(err)
}
fmt.Printf("Predicted: %s (%.1f%%)\n", res.Top.Label, res.Top.Probability*100)
```

## Testing

Run deterministic unit tests:

```bash
go test -v ./examples/semantic-search/classify
```

Run live integration tests against a real embedding `.litertlm` model:

```bash
LITERTLM_TEST_LIB=/abs/path/to/dist/lib \
LITERTLM_TEST_EMBED_MODEL=/abs/path/to/embeddinggemma-2-text-vision-440m.litertlm \
go test -v ./examples/semantic-search/classify -run TestLive_Classify_EndToEnd
```
