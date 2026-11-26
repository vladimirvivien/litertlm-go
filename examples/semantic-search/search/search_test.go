package search_test

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/vladimirvivien/litertlm-go/examples/semantic-search/search"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

type mockEmbedder struct {
	textVectors  map[string][]float32
	imageVectors map[int][]float32
	audioVectors map[int][]float32

	seenTexts  []string
	batchSizes []int
	embedErr   error
}

func (m *mockEmbedder) Embed(ctx context.Context, text string, _ ...litertlm.EmbeddingOption) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m.embedErr != nil {
		return nil, m.embedErr
	}
	m.seenTexts = append(m.seenTexts, text)
	if v, ok := m.textVectors[text]; ok {
		return slices.Clone(v), nil
	}
	return deterministicVec(text), nil
}

func (m *mockEmbedder) EmbedBatch(ctx context.Context, texts []string, opts ...litertlm.EmbeddingOption) ([][]float32, error) {
	m.batchSizes = append(m.batchSizes, len(texts))
	out := make([][]float32, len(texts))
	for i, txt := range texts {
		v, err := m.Embed(ctx, txt, opts...)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

func (m *mockEmbedder) EmbedMulti(ctx context.Context, parts []litertlm.Part, _ ...litertlm.EmbeddingOption) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m.embedErr != nil {
		return nil, m.embedErr
	}
	if len(parts) == 1 && parts[0].IsImage() {
		if v, ok := m.imageVectors[0]; ok {
			return slices.Clone(v), nil
		}
	}
	if len(parts) == 1 && parts[0].IsAudio() {
		if v, ok := m.audioVectors[0]; ok {
			return slices.Clone(v), nil
		}
	}
	return []float32{0.5, 0.5, 0, 0}, nil
}

func deterministicVec(s string) []float32 {
	low := strings.ToLower(s)
	switch {
	case strings.Contains(low, "parking") || strings.Contains(low, "car"):
		return []float32{0, 1, 0, 0}
	case strings.Contains(low, "table") || strings.Contains(low, "lamp"):
		return []float32{1, 0, 0, 0}
	case strings.Contains(low, "bread") || strings.Contains(low, "sourdough"):
		return []float32{0, 0, 1, 0}
	default:
		return []float32{0, 0, 0, 1}
	}
}

func TestDetectModality(t *testing.T) {
	cases := []struct {
		path string
		want search.Modality
		ok   bool
	}{
		{"notes.md", search.ModalityText, true},
		{"MAIN.GO", search.ModalityText, true},
		{"photo.JPG", search.ModalityImage, true},
		{"diagram.png", search.ModalityImage, true},
		{"clip.mp3", search.ModalityAudio, true},
		{"voice.WAV", search.ModalityAudio, true},
		{"archive.zip", "", false},
		{"noext", "", false},
	}
	for _, tc := range cases {
		got, ok := search.DetectModality(tc.path)
		if ok != tc.ok || got != tc.want {
			t.Errorf("DetectModality(%q) = (%q, %v), want (%q, %v)", tc.path, got, ok, tc.want, tc.ok)
		}
	}
}

func TestCosineSimilarity(t *testing.T) {
	if got := search.CosineSimilarity([]float32{1, 2, 3}, []float32{2, 4, 6}); math.Abs(got-1.0) > 1e-9 {
		t.Errorf("parallel vectors = %v, want 1.0", got)
	}
	if got := search.CosineSimilarity([]float32{1, 0}, []float32{0, 1}); got != 0 {
		t.Errorf("orthogonal vectors = %v, want 0", got)
	}
	if got := search.CosineSimilarity([]float32{1, 0}, []float32{-1, 0}); math.Abs(got+1.0) > 1e-9 {
		t.Errorf("opposite vectors = %v, want -1.0", got)
	}
	if got := search.CosineSimilarity(nil, nil); got != 0 {
		t.Errorf("empty vectors = %v, want 0", got)
	}
	if got := search.CosineSimilarity([]float32{0, 0}, []float32{1, 0}); got != 0 {
		t.Errorf("zero vector = %v, want 0", got)
	}
	if got := search.CosineSimilarity([]float32{1, 2}, []float32{1}); got != 0 {
		t.Errorf("mismatched lengths = %v, want 0", got)
	}
}

func TestIndex_AddBatchUpsertPrefixAndRank(t *testing.T) {
	emb := &mockEmbedder{}
	cfg := search.DefaultConfig()
	cfg.Model = "mock-embed"
	cfg.BatchSize = 2
	cfg.MaxPreviewRunes = 24
	idx := search.New(emb, cfg)

	err := idx.AddTexts(context.Background(),
		[]string{"doc-a", "doc-b", "doc-c"},
		[]string{
			"A small corner table\nwith a bright lamp and mug.",
			"Aerial parking lot with many cars",
			"Freshly baked sourdough bread",
		},
	)
	if err != nil {
		t.Fatalf("AddTexts: %v", err)
	}
	if len(emb.batchSizes) != 2 || emb.batchSizes[0] != 2 || emb.batchSizes[1] != 1 {
		t.Errorf("expected chunked batch sizes [2 1], got %v", emb.batchSizes)
	}
	if idx.Len() != 3 || idx.Dim() != 4 || idx.Model() != "mock-embed" {
		t.Fatalf("unexpected index metadata len=%d dim=%d model=%q", idx.Len(), idx.Dim(), idx.Model())
	}

	// Verify DocumentPrefix and preview collapsing/truncation.
	if !strings.HasPrefix(emb.seenTexts[0], search.DefaultDocumentPrefix) {
		t.Errorf("expected DocumentPrefix, got %q", emb.seenTexts[0])
	}
	entries := idx.Entries()
	if !strings.HasSuffix(entries[0].Preview, "...") || len([]rune(entries[0].Preview)) > 24 {
		t.Errorf("unexpected truncated preview %q", entries[0].Preview)
	}

	// Upsert doc-c to match table/lamp instead of bread.
	if err = idx.AddText(context.Background(), "doc-c", "table lamp guide"); err != nil {
		t.Fatalf("AddText upsert: %v", err)
	}
	if idx.Len() != 3 {
		t.Fatalf("expected upsert to keep Len=3, got %d", idx.Len())
	}

	matches, err := idx.QueryText(context.Background(), "lamp on table", 2)
	if err != nil {
		t.Fatalf("QueryText: %v", err)
	}
	if !strings.HasPrefix(emb.seenTexts[len(emb.seenTexts)-1], search.DefaultQueryPrefix) {
		t.Errorf("expected QueryPrefix on QueryText, got %q", emb.seenTexts[len(emb.seenTexts)-1])
	}
	if len(matches) != 2 || matches[0].Entry.ID != "doc-a" || matches[1].Entry.ID != "doc-c" {
		t.Errorf("unexpected tie-break top-2 ranking: %+v", matches)
	}
}

func TestIndex_DirMediaSaveLoadAndErrors(t *testing.T) {
	dir := t.TempDir()
	hiddenDir := filepath.Join(dir, ".git")
	subDir := filepath.Join(dir, "sub")
	if err := os.MkdirAll(hiddenDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(subDir, 0o755); err != nil {
		t.Fatal(err)
	}

	write := func(rel, content string) string {
		full := filepath.Join(dir, rel)
		if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		return full
	}

	write("table.md", "A corner table with a lamp")
	write("parking.txt", "Busy urban parking lot full of cars")
	write("empty.txt", "   \n  ")
	write("ignored.bin", "binary-bytes")
	write(".hidden.md", "should not index")
	write(".git/config.txt", "should not index hidden dir")
	write("huge.txt", strings.Repeat("x", 64))
	imgPath := write("sub/photo.png", "png-bytes")
	write("sub/clip.wav", "wav-bytes")

	emb := &mockEmbedder{
		imageVectors: map[int][]float32{0: {0, 1, 0, 0}},
		audioVectors: map[int][]float32{0: {0, 0, 1, 0}},
	}
	cfg := search.DefaultConfig()
	cfg.EnableVision = true
	cfg.EnableAudio = false
	cfg.MaxFileBytes = 50
	idx := search.New(emb, cfg)

	stats, err := idx.IndexDir(context.Background(), dir)
	if err != nil {
		t.Fatalf("IndexDir: %v", err)
	}
	if stats.Indexed != 3 || stats.TextCount != 2 || stats.ImageCount != 1 || stats.AudioCount != 0 || stats.Skipped < 4 {
		t.Fatalf("unexpected stats: %+v", stats)
	}

	// Cross-modal query using image path retrieves both image and parking.txt.
	matches, err := idx.QueryImage(context.Background(), imgPath, 2)
	if err != nil {
		t.Fatalf("QueryImage: %v", err)
	}
	if len(matches) != 2 || matches[0].Score < 0.99 {
		t.Fatalf("unexpected QueryImage matches: %+v", matches)
	}

	// Enable audio and add clip.wav explicitly via AddFile.
	cfgAudio := cfg
	cfgAudio.EnableAudio = true
	idxAudio := search.New(emb, cfgAudio)
	added, err := idxAudio.AddFile(context.Background(), filepath.Join(subDir, "clip.wav"))
	if err != nil || !added {
		t.Fatalf("AddFile(clip.wav) = (%v, %v), want (true, nil)", added, err)
	}
	if _, err = idxAudio.QueryAudio(context.Background(), filepath.Join(subDir, "clip.wav"), 1); err != nil {
		t.Fatalf("QueryAudio: %v", err)
	}
	if _, qErr := idx.QueryParts(context.Background(), []litertlm.Part{litertlm.Text("corner table")}, 1); qErr != nil {
		t.Fatalf("QueryParts: %v", qErr)
	}
	if _, dErr := idx.IndexDir(context.Background(), filepath.Join(dir, "missing-dir")); dErr == nil {
		t.Error("expected error indexing missing directory")
	}

	// Save & Load roundtrip.
	savePath := filepath.Join(t.TempDir(), "nested", "index.json")
	if err = idx.Save(savePath); err != nil {
		t.Fatalf("Save: %v", err)
	}
	loaded, err := search.Load(savePath, emb, search.DefaultConfig())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.Len() != idx.Len() || loaded.Dim() != idx.Dim() {
		t.Fatalf("loaded mismatch len=%d/%d dim=%d/%d", loaded.Len(), idx.Len(), loaded.Dim(), idx.Dim())
	}

	mrlCfg := search.DefaultConfig()
	mrlCfg.OutputDim = 256
	if _, mismatchErr := search.Load(savePath, emb, mrlCfg); mismatchErr == nil {
		t.Error("expected OutputDim mismatch error on Load")
	}

	// Dimension mismatch check on Rank.
	if _, rankErr := loaded.Rank([]float32{1, 2}, 1); rankErr == nil {
		t.Error("expected dimension mismatch error on Rank")
	}

	// Bad index version check.
	badPath := filepath.Join(t.TempDir(), "bad.json")
	badPayload, _ := json.Marshal(map[string]any{"version": 99, "dim": 4, "entries": []any{}})
	if err = os.WriteFile(badPath, badPayload, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, verErr := search.Load(badPath, emb, search.DefaultConfig()); verErr == nil {
		t.Error("expected error for unsupported index version")
	}

	// Nil embedder and cancelled context checks.
	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if err = idx.AddText(canceled, "x", "y"); !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", err)
	}
	nilIdx := search.New(nil, search.DefaultConfig())
	if err = nilIdx.AddText(context.Background(), "x", "y"); err == nil {
		t.Error("expected error when embedder is nil")
	}
}

func TestLive_Search_EndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test skipped in -short mode")
	}
	embedModel := os.Getenv("LITERTLM_TEST_EMBED_MODEL")
	if embedModel == "" {
		t.Skip("LITERTLM_TEST_EMBED_MODEL not set; skipping live search integration test")
	}
	libDir := os.Getenv("LITERTLM_TEST_LIB")
	if libDir == "" {
		libDir = os.Getenv("LITERTLM_LIB")
	}

	ctx := context.Background()
	if libDir != "" {
		if err := litertlm.Load(libDir, "cpu", ""); err != nil {
			t.Fatalf("Load: %v", err)
		}
	}
	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	caps, err := litertlm.InspectModel(embedModel)
	if err != nil {
		t.Fatalf("InspectModel: %v", err)
	}
	if caps.ModelType != litertlm.ModelTypeEmbedding {
		t.Fatalf("ModelType = %s, want Embedding", caps.ModelType)
	}

	client, err := litertlm.NewEmbeddingClient(ctx,
		litertlm.WithLib(libDir),
		litertlm.WithModel(embedModel),
		litertlm.WithBackend("cpu"),
	)
	if err != nil {
		t.Fatalf("NewEmbeddingClient: %v", err)
	}
	defer func() { _ = client.Close() }()

	cfg := search.DefaultConfig()
	cfg.Model = filepath.Base(embedModel)
	cfg.EnableVision = caps.SupportsVision()
	cfg.EnableAudio = caps.SupportsAudio()

	idx := search.New(client, cfg)
	testdataDir := filepath.Join("..", "..", "testdata")
	stats, err := idx.IndexDir(ctx, testdataDir)
	if err != nil {
		t.Fatalf("IndexDir(%q): %v", testdataDir, err)
	}
	t.Logf("indexed %+v (dim=%d)", stats, idx.Dim())

	parkingMatches, err := idx.QueryText(ctx, "overhead aerial photograph of a busy parking lot with rows of cars", 4)
	if err != nil {
		t.Fatalf("QueryText(parking): %v", err)
	}
	if len(parkingMatches) == 0 || !strings.HasPrefix(parkingMatches[0].Entry.ID, "img2.") {
		t.Errorf("expected img2.* top match for parking query, got %+v", parkingMatches)
	}

	tableMatches, err := idx.QueryText(ctx, "indoor corner table with lamp, reading glasses, and mug", 4)
	if err != nil {
		t.Fatalf("QueryText(table): %v", err)
	}
	if len(tableMatches) == 0 || !strings.HasPrefix(tableMatches[0].Entry.ID, "img1.") {
		t.Errorf("expected img1.* top match for table query, got %+v", tableMatches)
	}

	if caps.SupportsVision() {
		imgMatches, imgErr := idx.QueryImage(ctx, filepath.Join(testdataDir, "img2.png"), 4)
		if imgErr != nil {
			t.Fatalf("QueryImage(img2.png): %v", imgErr)
		}
		if len(imgMatches) < 2 || imgMatches[0].Entry.ID != "img2.png" || imgMatches[0].Score < 0.99 {
			t.Errorf("expected img2.png exact top match, got %+v", imgMatches)
		}
		var scoreImg1MD, scoreImg2MD float64
		for _, m := range imgMatches {
			switch m.Entry.ID {
			case "img1.md":
				scoreImg1MD = m.Score
			case "img2.md":
				scoreImg2MD = m.Score
			}
		}
		if scoreImg2MD <= scoreImg1MD {
			t.Errorf("expected cross-modal img2.png -> img2.md (%f) > img1.md (%f)", scoreImg2MD, scoreImg1MD)
		}
	}

	// Matryoshka 256-dim + persistence roundtrip.
	mrlCfg := cfg
	mrlCfg.OutputDim = 256
	mrlIdx := search.New(client, mrlCfg)
	if err = mrlIdx.AddText(ctx, "car", "Aerial view of parked cars"); err != nil {
		t.Fatalf("MRL AddText: %v", err)
	}
	if mrlIdx.Dim() != 256 {
		t.Errorf("MRL Dim = %d, want 256", mrlIdx.Dim())
	}

	savePath := filepath.Join(t.TempDir(), "index.json")
	if err = idx.Save(savePath); err != nil {
		t.Fatalf("Save: %v", err)
	}
	reloaded, err := search.Load(savePath, client, cfg)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if reloaded.Len() != idx.Len() || reloaded.Dim() != idx.Dim() {
		t.Errorf("reloaded mismatch len=%d/%d dim=%d/%d", reloaded.Len(), idx.Len(), reloaded.Dim(), idx.Dim())
	}
}
