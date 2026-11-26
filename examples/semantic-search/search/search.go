// Package search implements an in-process multimodal semantic index backed
// by litertlm.EmbeddingClient. Text, image, and audio files share one vector
// space, allowing cross-modal retrieval and similarity scoring.
package search

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

const (
	// IndexFormatVersion identifies the JSON schema written by Save and read by Load.
	IndexFormatVersion = 1

	// DefaultQueryPrefix is the EmbeddingGemma retrieval prompt prefix for queries.
	DefaultQueryPrefix = "task: search query | text: "

	// DefaultDocumentPrefix is the EmbeddingGemma retrieval prompt prefix for indexed text documents.
	DefaultDocumentPrefix = "task: search result | text: "

	// DefaultMaxFileBytes caps individual files skipped during directory indexing (10 MiB).
	DefaultMaxFileBytes int64 = 10 << 20

	// DefaultMaxPreviewRunes limits the single-line text snippet stored per Entry.
	DefaultMaxPreviewRunes = 120

	defaultBatchSize = 16
)

// Embedder abstracts the embedding operations required by Index.
// *litertlm.EmbeddingClient satisfies Embedder.
type Embedder interface {
	Embed(ctx context.Context, text string, opts ...litertlm.EmbeddingOption) ([]float32, error)
	EmbedBatch(ctx context.Context, texts []string, opts ...litertlm.EmbeddingOption) ([][]float32, error)
	EmbedMulti(ctx context.Context, parts []litertlm.Part, opts ...litertlm.EmbeddingOption) ([]float32, error)
}

// Modality classifies an indexed artifact by input type.
type Modality string

const (
	ModalityText  Modality = "text"
	ModalityImage Modality = "image"
	ModalityAudio Modality = "audio"
)

var modalityByExt = map[string]Modality{
	".txt":      ModalityText,
	".md":       ModalityText,
	".markdown": ModalityText,
	".go":       ModalityText,
	".json":     ModalityText,
	".yaml":     ModalityText,
	".yml":      ModalityText,
	".csv":      ModalityText,
	".toml":     ModalityText,
	".png":      ModalityImage,
	".jpg":      ModalityImage,
	".jpeg":     ModalityImage,
	".webp":     ModalityImage,
	".bmp":      ModalityImage,
	".wav":      ModalityAudio,
	".mp3":      ModalityAudio,
	".flac":     ModalityAudio,
	".ogg":      ModalityAudio,
}

// DetectModality maps a file path extension to a supported Modality.
func DetectModality(path string) (Modality, bool) {
	ext := strings.ToLower(filepath.Ext(path))
	m, ok := modalityByExt[ext]
	return m, ok
}

// Entry stores a single indexed item and its embedding vector.
type Entry struct {
	ID       string    `json:"id"`
	Path     string    `json:"path,omitempty"`
	Modality Modality  `json:"modality"`
	Preview  string    `json:"preview,omitempty"`
	Vector   []float32 `json:"vector"`
}

// Match pairs a ranked Entry with its cosine similarity score.
type Match struct {
	Entry Entry
	Score float64
}

// IndexStats summarizes work performed by IndexDir.
type IndexStats struct {
	Indexed    int
	Skipped    int
	TextCount  int
	ImageCount int
	AudioCount int
	Duration   time.Duration
}

// Config configures embedding options, prompt prefixes, and file scanning rules.
type Config struct {
	Model                string
	OutputDim            int
	Overflow             litertlm.OverflowStrategy
	VisionTokensPerImage int
	Normalize            *bool
	QueryPrefix          string
	DocumentPrefix       string
	EnableVision         bool
	EnableAudio          bool
	MaxFileBytes         int64
	MaxPreviewRunes      int
	BatchSize            int
}

// DefaultConfig returns recommended defaults for multimodal semantic search.
func DefaultConfig() Config {
	return Config{
		Overflow:        litertlm.OverflowChunkAndAverage,
		Normalize:       new(true),
		QueryPrefix:     DefaultQueryPrefix,
		DocumentPrefix:  DefaultDocumentPrefix,
		EnableVision:    true,
		EnableAudio:     false,
		MaxFileBytes:    DefaultMaxFileBytes,
		MaxPreviewRunes: DefaultMaxPreviewRunes,
		BatchSize:       defaultBatchSize,
	}
}

func (c Config) normalized() Config {
	if c.Normalize == nil {
		c.Normalize = new(true)
	}
	if c.MaxFileBytes <= 0 {
		c.MaxFileBytes = DefaultMaxFileBytes
	}
	if c.MaxPreviewRunes <= 0 {
		c.MaxPreviewRunes = DefaultMaxPreviewRunes
	}
	if c.BatchSize <= 0 {
		c.BatchSize = defaultBatchSize
	}
	return c
}

// EmbeddingOptions converts Config settings into per-call litertlm.EmbeddingOption values.
func (c Config) EmbeddingOptions() []litertlm.EmbeddingOption {
	norm := true
	if c.Normalize != nil {
		norm = *c.Normalize
	}
	opts := []litertlm.EmbeddingOption{
		litertlm.WithNormalize(norm),
		litertlm.WithOverflowStrategy(c.Overflow),
	}
	if c.OutputDim > 0 {
		opts = append(opts, litertlm.WithOutputSize(c.OutputDim))
	}
	if c.VisionTokensPerImage > 0 {
		opts = append(opts, litertlm.WithEmbeddingVisionTokensPerImage(c.VisionTokensPerImage))
	}
	return opts
}

type serializedIndex struct {
	Version int     `json:"version"`
	Model   string  `json:"model,omitempty"`
	Dim     int     `json:"dim"`
	Entries []Entry `json:"entries"`
}

// Index holds vectorized multimodal entries and ranks queries by cosine similarity.
type Index struct {
	embedder Embedder
	cfg      Config
	embOpts  []litertlm.EmbeddingOption

	mu      sync.RWMutex
	dim     int
	entries []Entry
	byID    map[string]int
}

// New creates an empty Index using embedder and cfg.
func New(embedder Embedder, cfg Config) *Index {
	normCfg := cfg.normalized()
	return &Index{
		embedder: embedder,
		cfg:      normCfg,
		embOpts:  normCfg.EmbeddingOptions(),
		dim:      normCfg.OutputDim,
		byID:     make(map[string]int),
	}
}

// Load reads a JSON index persisted by Save and attaches embedder for new additions and queries.
func Load(path string, embedder Embedder, cfg Config) (*Index, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("search: read index %q: %w", path, err)
	}
	var payload serializedIndex
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil, fmt.Errorf("search: parse index %q: %w", path, err)
	}
	if payload.Version != IndexFormatVersion {
		return nil, fmt.Errorf("search: unsupported index version %d (want %d)", payload.Version, IndexFormatVersion)
	}

	normCfg := cfg.normalized()
	normCfg.Model = cmp.Or(normCfg.Model, payload.Model)
	if normCfg.OutputDim > 0 && payload.Dim > 0 && normCfg.OutputDim != payload.Dim {
		return nil, fmt.Errorf("search: loaded index dimension %d does not match configured OutputDim %d", payload.Dim, normCfg.OutputDim)
	}

	idx := New(embedder, normCfg)
	if payload.Dim > 0 {
		idx.dim = payload.Dim
	}

	for i, entry := range payload.Entries {
		if entry.ID == "" {
			return nil, fmt.Errorf("search: entry %d has empty ID", i)
		}
		if len(entry.Vector) == 0 {
			return nil, fmt.Errorf("search: entry %q has empty vector", entry.ID)
		}
		if idx.dim == 0 {
			idx.dim = len(entry.Vector)
		} else if len(entry.Vector) != idx.dim {
			return nil, fmt.Errorf("search: entry %q vector dimension %d differs from index dimension %d",
				entry.ID, len(entry.Vector), idx.dim)
		}
		vecCopy := slices.Clone(entry.Vector)
		entry.Vector = vecCopy
		idx.upsertLocked(entry)
	}

	return idx, nil
}

// Save writes the index to path in formatted JSON.
func (idx *Index) Save(path string) error {
	if path == "" {
		return fmt.Errorf("search: index path is required")
	}

	idx.mu.RLock()
	entriesCopy := make([]Entry, len(idx.entries))
	for i, e := range idx.entries {
		entriesCopy[i] = Entry{
			ID:       e.ID,
			Path:     e.Path,
			Modality: e.Modality,
			Preview:  e.Preview,
			Vector:   slices.Clone(e.Vector),
		}
	}
	payload := serializedIndex{
		Version: IndexFormatVersion,
		Model:   idx.cfg.Model,
		Dim:     idx.dim,
		Entries: entriesCopy,
	}
	idx.mu.RUnlock()

	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("search: create index directory: %w", err)
		}
	}

	encoded, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return fmt.Errorf("search: encode index: %w", err)
	}
	encoded = append(encoded, '\n')
	if err := os.WriteFile(path, encoded, 0o644); err != nil {
		return fmt.Errorf("search: write index %q: %w", path, err)
	}
	return nil
}

// AddText embeds text with DocumentPrefix and inserts or updates id in the index.
func (idx *Index) AddText(ctx context.Context, id, text string) error {
	if err := idx.requireEmbedder(ctx); err != nil {
		return err
	}
	if strings.TrimSpace(id) == "" {
		return fmt.Errorf("search: AddText requires a non-empty ID")
	}
	if strings.TrimSpace(text) == "" {
		return fmt.Errorf("search: AddText requires non-empty text")
	}

	vec, err := idx.embedder.Embed(ctx, idx.cfg.DocumentPrefix+text, idx.embOpts...)
	if err != nil {
		return fmt.Errorf("search: embed %q: %w", id, err)
	}
	return idx.upsertAll([]Entry{{
		ID:       id,
		Modality: ModalityText,
		Preview:  previewText(text, idx.cfg.MaxPreviewRunes),
		Vector:   slices.Clone(vec),
	}})
}

// AddTexts batch-embeds ids and texts in chunks of Config.BatchSize and upserts them.
func (idx *Index) AddTexts(ctx context.Context, ids, texts []string) error {
	if err := idx.requireEmbedder(ctx); err != nil {
		return err
	}
	if len(ids) == 0 {
		return fmt.Errorf("search: AddTexts requires at least one item")
	}
	if len(ids) != len(texts) {
		return fmt.Errorf("search: AddTexts mismatched ids (%d) and texts (%d)", len(ids), len(texts))
	}

	pending := make([]textItem, 0, len(ids))
	for i := range ids {
		if strings.TrimSpace(ids[i]) == "" {
			return fmt.Errorf("search: AddTexts item %d has empty ID", i)
		}
		if strings.TrimSpace(texts[i]) == "" {
			return fmt.Errorf("search: AddTexts item %q has empty text", ids[i])
		}
		pending = append(pending, textItem{
			id:       ids[i],
			preview:  previewText(texts[i], idx.cfg.MaxPreviewRunes),
			prefixed: idx.cfg.DocumentPrefix + texts[i],
		})
	}
	return idx.embedTextBatch(ctx, pending)
}

type textItem struct {
	id       string
	path     string
	preview  string
	prefixed string
}

func (idx *Index) embedTextBatch(ctx context.Context, items []textItem) error {
	for start := 0; start < len(items); start += idx.cfg.BatchSize {
		if err := ctx.Err(); err != nil {
			return err
		}
		chunk := items[start:min(start+idx.cfg.BatchSize, len(items))]
		inputs := make([]string, len(chunk))
		for i, item := range chunk {
			inputs[i] = item.prefixed
		}
		vecs, err := idx.embedder.EmbedBatch(ctx, inputs, idx.embOpts...)
		if err != nil {
			return fmt.Errorf("search: batch embed text: %w", err)
		}
		if len(vecs) != len(chunk) {
			return fmt.Errorf("search: EmbedBatch returned %d vectors for %d inputs", len(vecs), len(chunk))
		}
		entries := make([]Entry, len(chunk))
		for i, item := range chunk {
			entries[i] = Entry{
				ID:       item.id,
				Path:     item.path,
				Modality: ModalityText,
				Preview:  item.preview,
				Vector:   slices.Clone(vecs[i]),
			}
		}
		if err := idx.upsertAll(entries); err != nil {
			return err
		}
	}
	return nil
}

// AddFile inspects path, embeds it according to its detected modality, and upserts it.
// Returns (false, nil) when the file is unsupported, disabled in Config, oversized, or empty text.
func (idx *Index) AddFile(ctx context.Context, path string) (bool, error) {
	if err := idx.requireEmbedder(ctx); err != nil {
		return false, err
	}
	return idx.addFileWithID(ctx, filepath.ToSlash(filepath.Clean(path)), path)
}

func (idx *Index) addFileWithID(ctx context.Context, id, path string) (bool, error) {
	modality, ok := idx.acceptFile(path)
	if !ok {
		return false, nil
	}

	switch modality {
	case ModalityText:
		item, accepted, err := idx.readTextItem(id, path)
		if err != nil || !accepted {
			return false, err
		}
		vec, err := idx.embedder.Embed(ctx, item.prefixed, idx.embOpts...)
		if err != nil {
			return false, fmt.Errorf("search: embed text file %q: %w", path, err)
		}
		if err := idx.upsertAll([]Entry{{
			ID:       item.id,
			Path:     item.path,
			Modality: ModalityText,
			Preview:  item.preview,
			Vector:   slices.Clone(vec),
		}}); err != nil {
			return false, err
		}
		return true, nil

	case ModalityImage:
		return idx.addMediaFile(ctx, id, path, ModalityImage, "[image]", litertlm.ImageFromFile)

	case ModalityAudio:
		return idx.addMediaFile(ctx, id, path, ModalityAudio, "[audio]", litertlm.AudioFromFile)

	default:
		return false, nil
	}
}

func (idx *Index) addMediaFile(
	ctx context.Context,
	id, path string,
	modality Modality,
	preview string,
	loader func(string) (litertlm.Part, error),
) (bool, error) {
	part, err := loader(path)
	if err != nil {
		return false, fmt.Errorf("search: load %s file %q: %w", modality, path, err)
	}
	vec, err := idx.embedder.EmbedMulti(ctx, []litertlm.Part{part}, idx.embOpts...)
	if err != nil {
		return false, fmt.Errorf("search: embed %s file %q: %w", modality, path, err)
	}
	if err := idx.upsertAll([]Entry{{
		ID:       id,
		Path:     path,
		Modality: modality,
		Preview:  preview,
		Vector:   slices.Clone(vec),
	}}); err != nil {
		return false, err
	}
	return true, nil
}

// IndexDir recursively indexes supported text, image, and audio files under root.
// Hidden entries and directories such as .git or node_modules are skipped.
func (idx *Index) IndexDir(ctx context.Context, root string) (IndexStats, error) {
	start := time.Now()
	var stats IndexStats
	if err := idx.requireEmbedder(ctx); err != nil {
		return stats, err
	}

	info, err := os.Stat(root)
	if err != nil {
		return stats, fmt.Errorf("search: stat root %q: %w", root, err)
	}
	if !info.IsDir() {
		return stats, fmt.Errorf("search: %q is not a directory", root)
	}

	var textItems []textItem
	var mediaFiles []struct {
		id       string
		path     string
		modality Modality
	}

	cleanRoot := filepath.Clean(root)
	walkErr := filepath.WalkDir(cleanRoot, func(path string, d fs.DirEntry, entryErr error) error {
		if entryErr != nil {
			return entryErr
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if d.IsDir() {
			if path != cleanRoot && shouldSkipDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasPrefix(d.Name(), ".") {
			stats.Skipped++
			return nil
		}

		modality, ok := idx.acceptFile(path)
		if !ok {
			stats.Skipped++
			return nil
		}

		rel, relErr := filepath.Rel(cleanRoot, path)
		if relErr != nil {
			rel = path
		}
		id := filepath.ToSlash(rel)

		switch modality {
		case ModalityText:
			item, accepted, readErr := idx.readTextItem(id, path)
			if readErr != nil {
				return readErr
			}
			if !accepted {
				stats.Skipped++
				return nil
			}
			textItems = append(textItems, item)
		case ModalityImage, ModalityAudio:
			mediaFiles = append(mediaFiles, struct {
				id       string
				path     string
				modality Modality
			}{id: id, path: path, modality: modality})
		}
		return nil
	})
	if walkErr != nil {
		return stats, walkErr
	}

	if len(textItems) > 0 {
		if err := idx.embedTextBatch(ctx, textItems); err != nil {
			return stats, err
		}
		stats.TextCount += len(textItems)
		stats.Indexed += len(textItems)
	}

	for _, mf := range mediaFiles {
		if err := ctx.Err(); err != nil {
			return stats, err
		}
		var (
			added bool
			mErr  error
		)
		if mf.modality == ModalityImage {
			added, mErr = idx.addMediaFile(ctx, mf.id, mf.path, ModalityImage, "[image]", litertlm.ImageFromFile)
		} else {
			added, mErr = idx.addMediaFile(ctx, mf.id, mf.path, ModalityAudio, "[audio]", litertlm.AudioFromFile)
		}
		if mErr != nil {
			return stats, mErr
		}
		if added {
			stats.Indexed++
			if mf.modality == ModalityImage {
				stats.ImageCount++
			} else {
				stats.AudioCount++
			}
		} else {
			stats.Skipped++
		}
	}

	stats.Duration = time.Since(start)
	return stats, nil
}

// QueryText embeds query using QueryPrefix and returns the top-K ranked matches.
func (idx *Index) QueryText(ctx context.Context, query string, topK int) ([]Match, error) {
	if err := idx.requireEmbedder(ctx); err != nil {
		return nil, err
	}
	if strings.TrimSpace(query) == "" {
		return nil, fmt.Errorf("search: query text is required")
	}
	vec, err := idx.embedder.Embed(ctx, idx.cfg.QueryPrefix+query, idx.embOpts...)
	if err != nil {
		return nil, fmt.Errorf("search: embed query: %w", err)
	}
	return idx.Rank(vec, topK)
}

// QueryImage embeds the image at path and returns the top-K ranked matches.
func (idx *Index) QueryImage(ctx context.Context, path string, topK int) ([]Match, error) {
	return idx.queryMediaFile(ctx, path, topK, litertlm.ImageFromFile)
}

// QueryAudio embeds the audio file at path and returns the top-K ranked matches.
func (idx *Index) QueryAudio(ctx context.Context, path string, topK int) ([]Match, error) {
	return idx.queryMediaFile(ctx, path, topK, litertlm.AudioFromFile)
}

func (idx *Index) queryMediaFile(
	ctx context.Context,
	path string,
	topK int,
	loader func(string) (litertlm.Part, error),
) ([]Match, error) {
	part, err := loader(path)
	if err != nil {
		return nil, fmt.Errorf("search: load query media %q: %w", path, err)
	}
	return idx.QueryParts(ctx, []litertlm.Part{part}, topK)
}

// QueryParts embeds multimodal parts and returns the top-K ranked matches.
func (idx *Index) QueryParts(ctx context.Context, parts []litertlm.Part, topK int) ([]Match, error) {
	if err := idx.requireEmbedder(ctx); err != nil {
		return nil, err
	}
	if len(parts) == 0 {
		return nil, fmt.Errorf("search: query requires at least one Part")
	}
	vec, err := idx.embedder.EmbedMulti(ctx, parts, idx.embOpts...)
	if err != nil {
		return nil, fmt.Errorf("search: embed multimodal query: %w", err)
	}
	return idx.Rank(vec, topK)
}

// Rank scores every indexed entry against queryVec by cosine similarity and returns
// the topK highest-scoring matches. Passing topK <= 0 returns all matches.
func (idx *Index) Rank(queryVec []float32, topK int) ([]Match, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(idx.entries) == 0 {
		return nil, nil
	}
	if len(queryVec) != idx.dim {
		return nil, fmt.Errorf("search: query vector dimension %d does not match index dimension %d", len(queryVec), idx.dim)
	}

	matches := make([]Match, len(idx.entries))
	for i, e := range idx.entries {
		matches[i] = Match{
			Entry: Entry{
				ID:       e.ID,
				Path:     e.Path,
				Modality: e.Modality,
				Preview:  e.Preview,
				Vector:   slices.Clone(e.Vector),
			},
			Score: CosineSimilarity(queryVec, e.Vector),
		}
	}

	sort.Slice(matches, func(i, j int) bool {
		if matches[i].Score != matches[j].Score {
			return matches[i].Score > matches[j].Score
		}
		return matches[i].Entry.ID < matches[j].Entry.ID
	})

	if topK > 0 && topK < len(matches) {
		matches = matches[:topK]
	}
	return matches, nil
}

// Entries returns a copy of all entries currently stored in the index.
func (idx *Index) Entries() []Entry {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	out := make([]Entry, len(idx.entries))
	for i, e := range idx.entries {
		out[i] = Entry{
			ID:       e.ID,
			Path:     e.Path,
			Modality: e.Modality,
			Preview:  e.Preview,
			Vector:   slices.Clone(e.Vector),
		}
	}
	return out
}

// Len reports the number of entries in the index.
func (idx *Index) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.entries)
}

// Dim returns the vector dimensionality of the index, or 0 if not yet determined.
func (idx *Index) Dim() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.dim
}

// Model returns the model identifier recorded in Config or loaded from disk.
func (idx *Index) Model() string {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.cfg.Model
}

// CosineSimilarity computes the cosine similarity between two equal-length vectors.
// Returns 0 when either slice is empty, norms are zero, or lengths differ.
func CosineSimilarity(a, b []float32) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		normA += x * x
		normB += y * y
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func (idx *Index) requireEmbedder(ctx context.Context) error {
	if idx == nil {
		return fmt.Errorf("search: index is nil")
	}
	if idx.embedder == nil {
		return fmt.Errorf("search: embedder is nil")
	}
	if ctx != nil {
		return ctx.Err()
	}
	return nil
}

func (idx *Index) acceptFile(path string) (Modality, bool) {
	modality, ok := DetectModality(path)
	if !ok {
		return "", false
	}
	if modality == ModalityImage && !idx.cfg.EnableVision {
		return "", false
	}
	if modality == ModalityAudio && !idx.cfg.EnableAudio {
		return "", false
	}
	fi, err := os.Stat(path)
	if err != nil || fi.IsDir() || fi.Size() <= 0 || fi.Size() > idx.cfg.MaxFileBytes {
		return "", false
	}
	return modality, true
}

func (idx *Index) readTextItem(id, path string) (textItem, bool, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return textItem{}, false, fmt.Errorf("search: read text file %q: %w", path, err)
	}
	content := strings.TrimSpace(string(raw))
	if content == "" {
		return textItem{}, false, nil
	}
	return textItem{
		id:       id,
		path:     path,
		preview:  previewText(content, idx.cfg.MaxPreviewRunes),
		prefixed: idx.cfg.DocumentPrefix + content,
	}, true, nil
}

func (idx *Index) upsertAll(entries []Entry) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for _, e := range entries {
		if len(e.Vector) == 0 {
			return fmt.Errorf("search: entry %q produced an empty embedding vector", e.ID)
		}
		if idx.dim == 0 {
			idx.dim = len(e.Vector)
		} else if len(e.Vector) != idx.dim {
			return fmt.Errorf("search: entry %q embedding dimension %d does not match index dimension %d",
				e.ID, len(e.Vector), idx.dim)
		}
	}
	for _, e := range entries {
		idx.upsertLocked(e)
	}
	return nil
}

func (idx *Index) upsertLocked(e Entry) {
	if existing, ok := idx.byID[e.ID]; ok {
		idx.entries[existing] = e
		return
	}
	idx.byID[e.ID] = len(idx.entries)
	idx.entries = append(idx.entries, e)
}

func shouldSkipDir(name string) bool {
	return strings.HasPrefix(name, ".") || name == "node_modules"
}

func previewText(s string, maxRunes int) string {
	collapsed := strings.Join(strings.Fields(s), " ")
	if maxRunes <= 0 || utf8.RuneCountInString(collapsed) <= maxRunes {
		return collapsed
	}
	if maxRunes <= 3 {
		return string([]rune(collapsed)[:maxRunes])
	}
	runes := []rune(collapsed)
	return string(runes[:maxRunes-3]) + "..."
}
