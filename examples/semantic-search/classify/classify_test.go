package classify_test

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/vladimirvivien/litertlm-go/examples/semantic-search/classify"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

type mockEmbedder struct {
	batchCalls int
	seenLabels []string
	seenInputs []string
}

func (m *mockEmbedder) Embed(ctx context.Context, text string, _ ...litertlm.EmbeddingOption) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.seenInputs = append(m.seenInputs, text)
	return vectorForText(text), nil
}

func (m *mockEmbedder) EmbedBatch(ctx context.Context, texts []string, _ ...litertlm.EmbeddingOption) ([][]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m.batchCalls++
	if m.batchCalls == 1 {
		m.seenLabels = slices.Clone(texts)
	} else {
		m.seenInputs = append(m.seenInputs, texts...)
	}
	out := make([][]float32, len(texts))
	for i, txt := range texts {
		out[i] = vectorForText(txt)
	}
	return out, nil
}

func (m *mockEmbedder) EmbedMulti(ctx context.Context, parts []litertlm.Part, _ ...litertlm.EmbeddingOption) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(parts) == 1 && parts[0].IsImage() {
		return []float32{0, 1, 0}, nil
	}
	if len(parts) == 1 && parts[0].IsAudio() {
		return []float32{0, 0, 1}, nil
	}
	return []float32{1, 0, 0}, nil
}

func vectorForText(s string) []float32 {
	low := strings.ToLower(s)
	switch {
	case strings.Contains(low, "table") || strings.Contains(low, "lamp"):
		return []float32{1, 0, 0}
	case strings.Contains(low, "parking") || strings.Contains(low, "car"):
		return []float32{0, 1, 0}
	default:
		return []float32{0, 0, 1}
	}
}

func TestSoftmaxProperties(t *testing.T) {
	if got := classify.Softmax(nil, 0.07); got != nil {
		t.Errorf("Softmax(nil) = %v, want nil", got)
	}

	sharp := classify.Softmax([]float64{0.9, 0.5, 0.1}, 0.05)
	flat := classify.Softmax([]float64{0.9, 0.5, 0.1}, 1.0)
	if sharp[0] <= flat[0] {
		t.Errorf("smaller temperature should produce sharper top probability (%v <= %v)", sharp[0], flat[0])
	}
	var sum float64
	for _, p := range sharp {
		sum += p
	}
	if math.Abs(sum-1.0) > 1e-9 {
		t.Errorf("probabilities sum = %v, want 1.0", sum)
	}

	// Large scaled inputs must remain finite without overflow.
	extreme := classify.Softmax([]float64{100, 99}, 0.01)
	if math.IsNaN(extreme[0]) || math.IsInf(extreme[0], 0) || math.Abs(extreme[0]+extreme[1]-1.0) > 1e-9 {
		t.Errorf("unexpected overflowed softmax: %v", extreme)
	}
}

func TestClassifier_UnitValidationAndScoring(t *testing.T) {
	ctx := context.Background()
	emb := &mockEmbedder{}

	if _, err := classify.NewFromNames(ctx, nil, []string{"a"}, classify.DefaultConfig()); err == nil {
		t.Error("expected error for nil embedder")
	}
	if _, err := classify.NewFromNames(ctx, emb, nil, classify.DefaultConfig()); err == nil {
		t.Error("expected error for empty labels")
	}
	if _, err := classify.NewFromNames(ctx, emb, []string{"dup", " dup "}, classify.DefaultConfig()); err == nil {
		t.Error("expected error for duplicate labels")
	}
	badCfg := classify.DefaultConfig()
	badCfg.Temperature = -0.1
	if _, err := classify.NewFromNames(ctx, emb, []string{"a"}, badCfg); err == nil {
		t.Error("expected error for negative temperature")
	}

	cfg := classify.DefaultConfig()
	cfg.LabelPrefix = "lbl:"
	cfg.InputPrefix = "in:"
	labels := []classify.Label{
		{Name: "indoor", Description: "a table and lamp"},
		{Name: "parking lot"},
		{Name: "culinary"},
	}
	clf, err := classify.New(ctx, emb, labels, cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if clf.Dim() != 3 || len(clf.Labels()) != 3 || emb.batchCalls != 1 {
		t.Fatalf("unexpected classifier init state dim=%d labels=%d batches=%d", clf.Dim(), len(clf.Labels()), emb.batchCalls)
	}
	if emb.seenLabels[0] != "lbl:a table and lamp" || emb.seenLabels[1] != "lbl:parking lot" {
		t.Errorf("unexpected prefixed labels: %v", emb.seenLabels)
	}

	// topK=2 truncates Scores while keeping full-label Softmax distribution.
	res, err := clf.ClassifyText(ctx, "reading lamp", 2)
	if err != nil {
		t.Fatalf("ClassifyText: %v", err)
	}
	if res.Top.Label != "indoor" || len(res.Scores) != 2 {
		t.Fatalf("unexpected topK=2 result: %+v", res)
	}
	if emb.seenInputs[len(emb.seenInputs)-1] != "in:reading lamp" {
		t.Errorf("expected InputPrefix on text input, got %q", emb.seenInputs[len(emb.seenInputs)-1])
	}
	if res.Scores[0].Probability+res.Scores[1].Probability >= 1.0 {
		t.Errorf("top-2 probabilities should sum strictly < 1 when a 3rd label exists: %+v", res.Scores)
	}

	// Ensure labels were NOT re-embedded when classifying multiple inputs.
	batchRes, err := clf.ClassifyTexts(ctx, []string{"parking cars", "sourdough bread"}, 0)
	if err != nil {
		t.Fatalf("ClassifyTexts: %v", err)
	}
	if len(batchRes) != 2 || batchRes[0].Top.Label != "parking lot" || batchRes[1].Top.Label != "culinary" {
		t.Errorf("unexpected batch results: %+v", batchRes)
	}
	if emb.batchCalls != 2 {
		t.Errorf("expected exactly 2 EmbedBatch calls (labels + ClassifyTexts), got %d", emb.batchCalls)
	}

	// ClassifyFile across text, image, audio, and unsupported extensions.
	dir := t.TempDir()
	txtFile := filepath.Join(dir, "doc.txt")
	imgFile := filepath.Join(dir, "shot.png")
	audFile := filepath.Join(dir, "clip.wav")
	binFile := filepath.Join(dir, "raw.bin")
	for p, body := range map[string]string{
		txtFile: "table lamp photo",
		imgFile: "png-bytes",
		audFile: "wav-bytes",
		binFile: "bin-bytes",
	} {
		if writeErr := os.WriteFile(p, []byte(body), 0o644); writeErr != nil {
			t.Fatal(writeErr)
		}
	}

	if txtRes, fErr := clf.ClassifyFile(ctx, txtFile, 0); fErr != nil || txtRes.Top.Label != "indoor" {
		t.Errorf("ClassifyFile(txt) = (%+v, %v), want indoor", txtRes, fErr)
	}
	if imgRes, fErr := clf.ClassifyFile(ctx, imgFile, 0); fErr != nil || imgRes.Top.Label != "parking lot" {
		t.Errorf("ClassifyFile(png) = (%+v, %v), want parking lot", imgRes, fErr)
	}
	if audRes, fErr := clf.ClassifyFile(ctx, audFile, 0); fErr != nil || audRes.Top.Label != "culinary" {
		t.Errorf("ClassifyFile(wav) = (%+v, %v), want culinary", audRes, fErr)
	}
	if _, fErr := clf.ClassifyFile(ctx, binFile, 0); fErr == nil {
		t.Error("expected error for unsupported .bin extension")
	}

	if _, sErr := clf.ScoreVector("bad", []float32{1, 0}, 0); sErr == nil {
		t.Error("expected dimension mismatch error on ScoreVector")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, cErr := clf.ClassifyText(canceled, "lamp", 0); !errors.Is(cErr, context.Canceled) {
		t.Errorf("expected context.Canceled, got %v", cErr)
	}
}

func TestLive_Classify_EndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test skipped in -short mode")
	}
	embedModel := os.Getenv("LITERTLM_TEST_EMBED_MODEL")
	if embedModel == "" {
		t.Skip("LITERTLM_TEST_EMBED_MODEL not set; skipping live classify integration test")
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

	labels := []string{
		"indoor corner table with lamp and glasses",
		"aerial photograph of a busy parking lot with cars",
		"ocean beach sunset",
	}
	clf, err := classify.NewFromNames(ctx, client, labels, classify.DefaultConfig())
	if err != nil {
		t.Fatalf("NewFromNames: %v", err)
	}

	txtRes, err := clf.ClassifyText(ctx, "many cars parked in horizontal rows seen from above", 0)
	if err != nil {
		t.Fatalf("ClassifyText: %v", err)
	}
	if txtRes.Top.Label != labels[1] {
		t.Errorf("ClassifyText top = %q, want %q (%+v)", txtRes.Top.Label, labels[1], txtRes.Scores)
	}
	var probSum float64
	for _, s := range txtRes.Scores {
		probSum += s.Probability
	}
	if math.Abs(probSum-1.0) > 1e-6 {
		t.Errorf("probability sum = %v, want ~1.0", probSum)
	}

	if caps.SupportsVision() {
		testdataDir := filepath.Join("..", "..", "testdata")
		res1, err := clf.ClassifyFile(ctx, filepath.Join(testdataDir, "img1.png"), 0)
		if err != nil {
			t.Fatalf("ClassifyFile(img1.png): %v", err)
		}
		if res1.Top.Label != labels[0] {
			t.Errorf("img1.png top = %q, want %q (%+v)", res1.Top.Label, labels[0], res1.Scores)
		}

		res2, err := clf.ClassifyFile(ctx, filepath.Join(testdataDir, "img2.png"), 0)
		if err != nil {
			t.Fatalf("ClassifyFile(img2.png): %v", err)
		}
		if res2.Top.Label != labels[1] {
			t.Errorf("img2.png top = %q, want %q (%+v)", res2.Top.Label, labels[1], res2.Scores)
		}
	}
}
