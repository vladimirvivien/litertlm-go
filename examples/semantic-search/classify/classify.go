// Package classify implements zero-shot text and multimodal classification by
// ranking cosine similarity against pre-embedded candidate labels and converting
// similarities into temperature-scaled softmax probabilities.
package classify

import (
	"context"
	"fmt"
	"math"
	"os"
	"slices"
	"sort"
	"strings"

	"github.com/vladimirvivien/litertlm-go/examples/semantic-search/search"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

// DefaultTemperature scales cosine similarities in Softmax when Config.Temperature is 0.
const DefaultTemperature = 0.07

// Embedder abstracts the embedding operations required by Classifier.
// *litertlm.EmbeddingClient satisfies Embedder.
type Embedder interface {
	Embed(ctx context.Context, text string, opts ...litertlm.EmbeddingOption) ([]float32, error)
	EmbedBatch(ctx context.Context, texts []string, opts ...litertlm.EmbeddingOption) ([][]float32, error)
	EmbedMulti(ctx context.Context, parts []litertlm.Part, opts ...litertlm.EmbeddingOption) ([]float32, error)
}

// Label defines one candidate class. When Description is non-empty, it is
// embedded instead of Name while preserving Name in Score results.
type Label struct {
	Name        string
	Description string
}

// Score holds similarity and probability for a single candidate label.
type Score struct {
	Label       string
	Similarity  float64
	Probability float64
}

// Result contains the predicted top label and ranked label scores for an input.
type Result struct {
	Input  string
	Top    Score
	Scores []Score
}

// Config configures embedding options, temperature scaling, and optional prompt prefixes.
type Config struct {
	OutputDim            int
	Overflow             litertlm.OverflowStrategy
	VisionTokensPerImage int
	Normalize            *bool
	Temperature          float64
	LabelPrefix          string
	InputPrefix          string
}

// DefaultConfig returns recommended defaults for zero-shot classification.
func DefaultConfig() Config {
	return Config{
		Overflow:    litertlm.OverflowChunkAndAverage,
		Normalize:   new(true),
		Temperature: DefaultTemperature,
	}
}

func (c Config) normalized() (Config, error) {
	if c.Temperature < 0 {
		return c, fmt.Errorf("classify: temperature must be non-negative (got %v)", c.Temperature)
	}
	if c.Temperature == 0 {
		c.Temperature = DefaultTemperature
	}
	if c.Normalize == nil {
		c.Normalize = new(true)
	}
	return c, nil
}

// EmbeddingOptions converts Config into per-call litertlm.EmbeddingOption values.
func (c Config) EmbeddingOptions() []litertlm.EmbeddingOption {
	return search.Config{
		OutputDim:            c.OutputDim,
		Overflow:             c.Overflow,
		VisionTokensPerImage: c.VisionTokensPerImage,
		Normalize:            c.Normalize,
	}.EmbeddingOptions()
}

// Classifier holds pre-embedded candidate labels and classifies inputs by cosine similarity.
type Classifier struct {
	embedder  Embedder
	cfg       Config
	embOpts   []litertlm.EmbeddingOption
	labels    []Label
	labelVecs [][]float32
	dim       int
}

// New embeds every candidate label once and returns a Classifier ready for inference.
func New(ctx context.Context, embedder Embedder, labels []Label, cfg Config) (*Classifier, error) {
	if embedder == nil {
		return nil, fmt.Errorf("classify: embedder is nil")
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	normCfg, err := cfg.normalized()
	if err != nil {
		return nil, err
	}
	if len(labels) == 0 {
		return nil, fmt.Errorf("classify: at least one candidate label is required")
	}

	cleaned := make([]Label, len(labels))
	texts := make([]string, len(labels))
	seen := make(map[string]struct{}, len(labels))
	for i, lbl := range labels {
		name := strings.TrimSpace(lbl.Name)
		if name == "" {
			return nil, fmt.Errorf("classify: label %d has empty name", i)
		}
		if _, dup := seen[name]; dup {
			return nil, fmt.Errorf("classify: duplicate label %q", name)
		}
		seen[name] = struct{}{}

		desc := strings.TrimSpace(lbl.Description)
		phrase := desc
		if phrase == "" {
			phrase = name
		}
		cleaned[i] = Label{Name: name, Description: desc}
		texts[i] = normCfg.LabelPrefix + phrase
	}

	embOpts := normCfg.EmbeddingOptions()
	vecs, err := embedder.EmbedBatch(ctx, texts, embOpts...)
	if err != nil {
		return nil, fmt.Errorf("classify: embed candidate labels: %w", err)
	}
	if len(vecs) != len(cleaned) {
		return nil, fmt.Errorf("classify: EmbedBatch returned %d vectors for %d labels", len(vecs), len(cleaned))
	}

	dim := normCfg.OutputDim
	copiedVecs := make([][]float32, len(vecs))
	for i, vec := range vecs {
		if len(vec) == 0 {
			return nil, fmt.Errorf("classify: label %q produced empty vector", cleaned[i].Name)
		}
		if dim == 0 {
			dim = len(vec)
		} else if len(vec) != dim {
			return nil, fmt.Errorf("classify: label %q vector dimension %d differs from expected %d",
				cleaned[i].Name, len(vec), dim)
		}
		copiedVecs[i] = slices.Clone(vec)
	}

	return &Classifier{
		embedder:  embedder,
		cfg:       normCfg,
		embOpts:   embOpts,
		labels:    cleaned,
		labelVecs: copiedVecs,
		dim:       dim,
	}, nil
}

// NewFromNames constructs a Classifier from plain label name strings.
func NewFromNames(ctx context.Context, embedder Embedder, names []string, cfg Config) (*Classifier, error) {
	labels := make([]Label, len(names))
	for i, name := range names {
		labels[i] = Label{Name: name}
	}
	return New(ctx, embedder, labels, cfg)
}

// ClassifyText scores a single text input against the cached labels.
func (c *Classifier) ClassifyText(ctx context.Context, input string, topK int) (Result, error) {
	if err := c.requireReady(ctx); err != nil {
		return Result{}, err
	}
	if strings.TrimSpace(input) == "" {
		return Result{}, fmt.Errorf("classify: text input is empty")
	}
	vec, err := c.embedder.Embed(ctx, c.cfg.InputPrefix+input, c.embOpts...)
	if err != nil {
		return Result{}, fmt.Errorf("classify: embed text: %w", err)
	}
	return c.ScoreVector(input, vec, topK)
}

// ClassifyTexts batch-embeds multiple text inputs and scores each against the cached labels.
func (c *Classifier) ClassifyTexts(ctx context.Context, inputs []string, topK int) ([]Result, error) {
	if err := c.requireReady(ctx); err != nil {
		return nil, err
	}
	if len(inputs) == 0 {
		return nil, fmt.Errorf("classify: ClassifyTexts requires at least one input")
	}
	prefixed := make([]string, len(inputs))
	for i, in := range inputs {
		if strings.TrimSpace(in) == "" {
			return nil, fmt.Errorf("classify: text input %d is empty", i)
		}
		prefixed[i] = c.cfg.InputPrefix + in
	}
	vecs, err := c.embedder.EmbedBatch(ctx, prefixed, c.embOpts...)
	if err != nil {
		return nil, fmt.Errorf("classify: batch embed texts: %w", err)
	}
	if len(vecs) != len(inputs) {
		return nil, fmt.Errorf("classify: EmbedBatch returned %d vectors for %d inputs", len(vecs), len(inputs))
	}
	results := make([]Result, len(inputs))
	for i, vec := range vecs {
		res, err := c.ScoreVector(inputs[i], vec, topK)
		if err != nil {
			return nil, err
		}
		results[i] = res
	}
	return results, nil
}

// ClassifyImage embeds an image file and scores it against the cached labels.
func (c *Classifier) ClassifyImage(ctx context.Context, path string, topK int) (Result, error) {
	return c.classifyMediaFile(ctx, path, topK, litertlm.ImageFromFile)
}

// ClassifyAudio embeds an audio file and scores it against the cached labels.
func (c *Classifier) ClassifyAudio(ctx context.Context, path string, topK int) (Result, error) {
	return c.classifyMediaFile(ctx, path, topK, litertlm.AudioFromFile)
}

func (c *Classifier) classifyMediaFile(
	ctx context.Context,
	path string,
	topK int,
	loader func(string) (litertlm.Part, error),
) (Result, error) {
	if err := c.requireReady(ctx); err != nil {
		return Result{}, err
	}
	part, err := loader(path)
	if err != nil {
		return Result{}, fmt.Errorf("classify: load media %q: %w", path, err)
	}
	return c.ClassifyParts(ctx, path, []litertlm.Part{part}, topK)
}

// ClassifyFile detects whether path is text, image, or audio and classifies its contents.
func (c *Classifier) ClassifyFile(ctx context.Context, path string, topK int) (Result, error) {
	if err := c.requireReady(ctx); err != nil {
		return Result{}, err
	}
	modality, ok := search.DetectModality(path)
	if !ok {
		return Result{}, fmt.Errorf("classify: unsupported file type for %q", path)
	}

	switch modality {
	case search.ModalityText:
		raw, err := os.ReadFile(path)
		if err != nil {
			return Result{}, fmt.Errorf("classify: read text file %q: %w", path, err)
		}
		content := strings.TrimSpace(string(raw))
		if content == "" {
			return Result{}, fmt.Errorf("classify: text file %q is empty", path)
		}
		vec, err := c.embedder.Embed(ctx, c.cfg.InputPrefix+content, c.embOpts...)
		if err != nil {
			return Result{}, fmt.Errorf("classify: embed text file %q: %w", path, err)
		}
		return c.ScoreVector(path, vec, topK)
	case search.ModalityImage:
		return c.ClassifyImage(ctx, path, topK)
	case search.ModalityAudio:
		return c.ClassifyAudio(ctx, path, topK)
	default:
		return Result{}, fmt.Errorf("classify: unsupported modality %q", modality)
	}
}

// ClassifyParts embeds multimodal parts and scores them against the cached labels.
func (c *Classifier) ClassifyParts(ctx context.Context, inputName string, parts []litertlm.Part, topK int) (Result, error) {
	if err := c.requireReady(ctx); err != nil {
		return Result{}, err
	}
	if len(parts) == 0 {
		return Result{}, fmt.Errorf("classify: ClassifyParts requires at least one Part")
	}
	vec, err := c.embedder.EmbedMulti(ctx, parts, c.embOpts...)
	if err != nil {
		return Result{}, fmt.Errorf("classify: embed multimodal input: %w", err)
	}
	return c.ScoreVector(inputName, vec, topK)
}

// ScoreVector scores a precomputed embedding vector against every cached label,
// applies Softmax across the full label set, and truncates Scores to topK.
func (c *Classifier) ScoreVector(input string, vec []float32, topK int) (Result, error) {
	if c == nil {
		return Result{}, fmt.Errorf("classify: classifier is nil")
	}
	if len(vec) != c.dim {
		return Result{}, fmt.Errorf("classify: vector dimension %d does not match classifier dimension %d", len(vec), c.dim)
	}

	sims := make([]float64, len(c.labels))
	for i, lblVec := range c.labelVecs {
		sims[i] = search.CosineSimilarity(vec, lblVec)
	}
	probs := Softmax(sims, c.cfg.Temperature)

	scores := make([]Score, len(c.labels))
	for i, lbl := range c.labels {
		scores[i] = Score{
			Label:       lbl.Name,
			Similarity:  sims[i],
			Probability: probs[i],
		}
	}

	sort.Slice(scores, func(i, j int) bool {
		if scores[i].Similarity != scores[j].Similarity {
			return scores[i].Similarity > scores[j].Similarity
		}
		return scores[i].Label < scores[j].Label
	})

	top := scores[0]
	if topK > 0 && topK < len(scores) {
		scores = scores[:topK]
	}
	return Result{
		Input:  input,
		Top:    top,
		Scores: scores,
	}, nil
}

// Labels returns a copy of the configured candidate labels.
func (c *Classifier) Labels() []Label {
	if c == nil {
		return nil
	}
	return slices.Clone(c.labels)
}

// Dim returns the embedding vector dimensionality of the cached labels.
func (c *Classifier) Dim() int {
	if c == nil {
		return 0
	}
	return c.dim
}

// Softmax converts similarity scores into a probability distribution scaled by temperature.
func Softmax(values []float64, temperature float64) []float64 {
	if len(values) == 0 {
		return nil
	}
	if temperature <= 0 {
		temperature = DefaultTemperature
	}

	scaled := make([]float64, len(values))
	maxVal := math.Inf(-1)
	for i, v := range values {
		s := v / temperature
		scaled[i] = s
		if s > maxVal {
			maxVal = s
		}
	}

	out := make([]float64, len(values))
	var sum float64
	for i, s := range scaled {
		e := math.Exp(s - maxVal)
		out[i] = e
		sum += e
	}
	if sum == 0 {
		uniform := 1.0 / float64(len(values))
		for i := range out {
			out[i] = uniform
		}
		return out
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func (c *Classifier) requireReady(ctx context.Context) error {
	if c == nil || c.embedder == nil {
		return fmt.Errorf("classify: classifier is not initialized")
	}
	if ctx != nil {
		return ctx.Err()
	}
	return nil
}
