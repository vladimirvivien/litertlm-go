package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/vladimirvivien/litertlm-go/examples/semantic-search/classify"
	"github.com/vladimirvivien/litertlm-go/examples/semantic-search/search"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

type cliMockEmbedder struct{}

func (m *cliMockEmbedder) Embed(_ context.Context, text string, _ ...litertlm.EmbeddingOption) ([]float32, error) {
	low := strings.ToLower(text)
	if strings.Contains(low, "car") || strings.Contains(low, "parking") {
		return []float32{0, 1, 0}, nil
	}
	if strings.Contains(low, "table") || strings.Contains(low, "lamp") {
		return []float32{1, 0, 0}, nil
	}
	return []float32{0, 0, 1}, nil
}

func (m *cliMockEmbedder) EmbedBatch(ctx context.Context, texts []string, opts ...litertlm.EmbeddingOption) ([][]float32, error) {
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

func (m *cliMockEmbedder) EmbedMulti(_ context.Context, parts []litertlm.Part, _ ...litertlm.EmbeddingOption) ([]float32, error) {
	if len(parts) > 0 && parts[0].IsImage() {
		return []float32{1, 0, 0}, nil
	}
	return []float32{0, 0, 1}, nil
}

func TestRunDispatchHelpAndErrors(t *testing.T) {
	var out, errOut bytes.Buffer

	if code := run(context.Background(), []string{"help"}, strings.NewReader(""), &out, &errOut); code != 0 {
		t.Fatalf("help exit code = %d, want 0", code)
	}
	if !strings.Contains(out.String(), "semantic-search search") || !strings.Contains(out.String(), "semantic-search classify") {
		t.Errorf("unexpected help output: %s", out.String())
	}

	out.Reset()
	errOut.Reset()
	if code := run(context.Background(), nil, strings.NewReader(""), &out, &errOut); code != 2 {
		t.Errorf("empty args exit code = %d, want 2", code)
	}

	errOut.Reset()
	if code := run(context.Background(), []string{"bogus"}, strings.NewReader(""), &out, &errOut); code != 2 {
		t.Errorf("unknown subcommand exit code = %d, want 2", code)
	}

	out.Reset()
	errOut.Reset()
	if code := run(context.Background(), []string{"search", "-h"}, strings.NewReader(""), &out, &errOut); code != 0 {
		t.Errorf("search -h exit code = %d, want 0", code)
	}

	errOut.Reset()
	if code := run(context.Background(), []string{"search"}, strings.NewReader(""), &out, &errOut); code != 2 {
		t.Errorf("search without -dir/-index exit code = %d, want 2", code)
	}

	errOut.Reset()
	if code := run(context.Background(), []string{"classify", "-labels", "a,b"}, strings.NewReader(""), &out, &errOut); code != 2 {
		t.Errorf("classify without inputs exit code = %d, want 2", code)
	}
}

func TestHelpersAndValidation(t *testing.T) {
	labels, err := collectLabels(" indoor , outdoor, indoor ", []string{"food,drink", ""})
	if err != nil {
		t.Fatalf("collectLabels: %v", err)
	}
	gotNames := make([]string, len(labels))
	for i, l := range labels {
		gotNames[i] = l.Name
	}
	wantNames := []string{"indoor", "outdoor", "food,drink"}
	if !slices.Equal(gotNames, wantNames) {
		t.Errorf("collectLabels = %v, want %v", gotNames, wantNames)
	}
	if _, err := collectLabels(" , ", nil); err == nil {
		t.Error("expected error for empty labels")
	}

	for raw, want := range map[string]litertlm.OverflowStrategy{
		"":                  litertlm.OverflowChunkAndAverage,
		"chunk":             litertlm.OverflowChunkAndAverage,
		"chunk_and_average": litertlm.OverflowChunkAndAverage,
		"truncate":          litertlm.OverflowTruncate,
		"error":             litertlm.OverflowError,
	} {
		got, err := parseOverflow(raw)
		if err != nil || got != want {
			t.Errorf("parseOverflow(%q) = (%v, %v), want (%v, nil)", raw, got, err, want)
		}
	}
	if _, err := parseOverflow("invalid"); err == nil {
		t.Error("expected error on unknown overflow strategy")
	}

	if err := validateSearchArgs("", "", "q", "", ""); err == nil {
		t.Error("expected error when both -dir and -index empty")
	}
	if err := validateSearchArgs("d", "", "q", "img.png", ""); err == nil {
		t.Error("expected error when multiple query flags set")
	}
	if err := validateSearchArgs("d", "", "q", "", ""); err != nil {
		t.Errorf("unexpected validateSearchArgs error: %v", err)
	}

	if got := renderBar(0.5, 4); got != "[##..]" {
		t.Errorf("renderBar(0.5, 4) = %q, want [##..]", got)
	}
	if got := renderBar(2.0, 4); got != "[####]" {
		t.Errorf("renderBar(2.0, 4) = %q, want [####]", got)
	}
	if got := unquotePath(`"hello world.png"`); got != "hello world.png" {
		t.Errorf("unquotePath = %q", got)
	}
}

func TestSearchREPLAndFormatting(t *testing.T) {
	emb := &cliMockEmbedder{}
	idx := search.New(emb, search.Config{Model: "mock"})
	if err := idx.AddTexts(context.Background(),
		[]string{"room.md", "lot.md"},
		[]string{"corner table with lamp", "parking lot with cars"},
	); err != nil {
		t.Fatal(err)
	}

	imgPath := filepath.Join(t.TempDir(), "sample.png")
	if err := os.WriteFile(imgPath, []byte("png"), 0o644); err != nil {
		t.Fatal(err)
	}

	script := strings.Join([]string{
		":help",
		":stats",
		"parking lot cars",
		":image \"" + imgPath + "\"",
		":unknown",
		":quit",
	}, "\n") + "\n"

	var out bytes.Buffer
	if err := runSearchREPL(context.Background(), idx, 2, strings.NewReader(script), &out); err != nil {
		t.Fatalf("runSearchREPL: %v", err)
	}

	got := out.String()
	for _, want := range []string{
		"Commands:",
		"Index stats: model=mock entries=2 dim=3 (text=2 image=0 audio=0)",
		`Results for text query "parking lot cars"`,
		"lot.md",
		"Results for image query",
		"room.md",
		`unknown command ":unknown"`,
		"bye",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("REPL output missing %q\n--- output ---\n%s", want, got)
		}
	}

	var clfOut bytes.Buffer
	res := classify.Result{
		Input: "shot.png",
		Top:   classify.Score{Label: "indoor", Similarity: 0.42, Probability: 0.85},
		Scores: []classify.Score{
			{Label: "indoor", Similarity: 0.42, Probability: 0.85},
			{Label: "outdoor", Similarity: 0.12, Probability: 0.15},
		},
	}
	printClassifyResult(&clfOut, res, 0)
	if !strings.Contains(clfOut.String(), "Top: indoor") || !strings.Contains(clfOut.String(), "85.0%") {
		t.Errorf("unexpected formatted classify output: %s", clfOut.String())
	}
}
