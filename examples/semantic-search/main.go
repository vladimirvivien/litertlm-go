// semantic-search demonstrates two multimodal embedding workflows powered by
// litertlm.EmbeddingClient: cross-modal retrieval over mixed files (`search`
// subcommand) and zero-shot classification (`classify` subcommand).
//
// See README.md in this directory for prerequisites and usage.
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/vladimirvivien/litertlm-go/examples/semantic-search/classify"
	"github.com/vladimirvivien/litertlm-go/examples/semantic-search/search"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	os.Exit(run(ctx, os.Args[1:], os.Stdin, os.Stdout, os.Stderr))
}

func run(ctx context.Context, args []string, in io.Reader, out, errOut io.Writer) int {
	if len(args) == 0 {
		printTopUsage(errOut)
		return 2
	}

	switch args[0] {
	case "-h", "--help", "help":
		printTopUsage(out)
		return 0
	case "search":
		code, err := runSearchCmd(ctx, args[1:], in, out, errOut)
		if err != nil {
			fmt.Fprintf(errOut, "search: %v\n", err)
		}
		return code
	case "classify":
		code, err := runClassifyCmd(ctx, args[1:], out, errOut)
		if err != nil {
			fmt.Fprintf(errOut, "classify: %v\n", err)
		}
		return code
	default:
		fmt.Fprintf(errOut, "unknown subcommand %q\n\n", args[0])
		printTopUsage(errOut)
		return 2
	}
}

func printTopUsage(w io.Writer) {
	_, _ = fmt.Fprintln(w, `Usage:
  semantic-search search   [flags]               Index files/folders and run text/image/audio retrieval
  semantic-search classify [flags] [files...]    Zero-shot classification of files or text inputs

Run a subcommand with -h for its flags:
  go run ./examples/semantic-search search -h
  go run ./examples/semantic-search classify -h`)
}

type commonFlags struct {
	model         string
	getModel      string
	libPath       string
	getLib        string
	backend       string
	visionBackend string
	audioBackend  string
	dim           int
	overflow      string
	normalize     bool
	visionTokens  int
}

func bindCommonFlags(fs *flag.FlagSet) *commonFlags {
	f := &commonFlags{}
	fs.StringVar(&f.model, "model", "", "path to an embedding .litertlm model file (or set LITERTLM_MODEL)")
	fs.StringVar(&f.getModel, "get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/embeddinggemma-2-text-vision-440m-litert-lm)")
	fs.StringVar(&f.libPath, "lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	fs.StringVar(&f.getLib, "get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.19.1)")
	fs.StringVar(&f.backend, "backend", "cpu", "inference backend (cpu | gpu | npu)")
	fs.StringVar(&f.visionBackend, "vision-backend", "", "optional vision encoder backend override (cpu | gpu | npu)")
	fs.StringVar(&f.audioBackend, "audio-backend", "", "optional audio encoder backend override (cpu | gpu | npu)")
	fs.IntVar(&f.dim, "dim", 0, "optional Matryoshka truncated output dimension (0 = full model dimension)")
	fs.StringVar(&f.overflow, "overflow", "chunk", "overflow strategy (chunk | truncate | error)")
	fs.BoolVar(&f.normalize, "normalize", true, "L2-normalize output embeddings")
	fs.IntVar(&f.visionTokens, "vision-tokens", 0, "optional per-image vision token budget override (0 = model default)")
	return f
}

func (f *commonFlags) parseOverflow() (litertlm.OverflowStrategy, error) {
	return parseOverflow(f.overflow)
}

func (f *commonFlags) openClient(ctx context.Context) (*litertlm.EmbeddingClient, *litertlm.ModelCapabilities, string, error) {
	resolvedLib := f.libPath
	if f.getLib != "" {
		staged, err := litertlm.FetchLib(ctx, "", "", f.getLib)
		if err != nil {
			return nil, nil, "", fmt.Errorf("fetch library: %w", err)
		}
		resolvedLib = staged
	}

	resolvedModel := f.model
	if resolvedModel == "" {
		resolvedModel = os.Getenv("LITERTLM_MODEL")
	}
	if f.getModel != "" {
		staged, err := litertlm.FetchModel(ctx, f.getModel)
		if err != nil {
			return nil, nil, "", fmt.Errorf("fetch model: %w", err)
		}
		resolvedModel = staged
	}
	if resolvedModel == "" {
		return nil, nil, "", fmt.Errorf("-model or -get-model (or LITERTLM_MODEL env) is required")
	}

	if resolvedLib != "" {
		if err := litertlm.Load(resolvedLib, f.backend, ""); err != nil {
			return nil, nil, "", fmt.Errorf("load library: %w", err)
		}
		litertlm.SetMinLogLevel(litertlm.LogQuiet)
	}

	caps, err := litertlm.InspectModel(resolvedModel)
	if err != nil {
		return nil, nil, "", fmt.Errorf("inspect model: %w", err)
	}
	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	if caps.ModelType != litertlm.ModelTypeEmbedding {
		return nil, nil, "", fmt.Errorf("model %s has ModelType=%s; an Embedding .litertlm model is required",
			resolvedModel, caps.ModelType)
	}

	opts := []litertlm.Option{
		litertlm.WithModel(resolvedModel),
		litertlm.WithBackend(f.backend),
	}
	if resolvedLib != "" {
		opts = append(opts, litertlm.WithLib(resolvedLib))
	}
	if f.visionBackend != "" {
		opts = append(opts, litertlm.WithVisionBackend(f.visionBackend))
	}
	if f.audioBackend != "" {
		opts = append(opts, litertlm.WithAudioBackend(f.audioBackend))
	}

	client, err := litertlm.NewEmbeddingClient(ctx, opts...)
	if err != nil {
		return nil, nil, "", err
	}
	return client, caps, resolvedModel, nil
}

func runSearchCmd(ctx context.Context, args []string, in io.Reader, out, errOut io.Writer) (int, error) {
	fs := flag.NewFlagSet("search", flag.ContinueOnError)
	fs.SetOutput(errOut)
	cf := bindCommonFlags(fs)
	dir := fs.String("dir", "", "directory to scan and embed (requires -dir or an existing -index)")
	indexPath := fs.String("index", "", "path to JSON index file (save after indexing -dir, or load if -dir is omitted)")
	queryText := fs.String("query", "", "one-shot text query")
	queryImage := fs.String("query-image", "", "one-shot image query file path")
	queryAudio := fs.String("query-audio", "", "one-shot audio query file path")
	topK := fs.Int("top", 5, "maximum number of ranked results to display (0 = all)")
	queryPrefix := fs.String("query-prefix", search.DefaultQueryPrefix, "prompt prefix prepended to text queries")
	docPrefix := fs.String("doc-prefix", search.DefaultDocumentPrefix, "prompt prefix prepended to indexed text documents")

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0, nil
		}
		return 2, nil
	}

	if err := validateSearchArgs(*dir, *indexPath, *queryText, *queryImage, *queryAudio); err != nil {
		return 2, err
	}
	strategy, err := cf.parseOverflow()
	if err != nil {
		return 2, err
	}

	client, caps, resolvedModel, err := cf.openClient(ctx)
	if err != nil {
		return 1, err
	}
	defer func() { _ = client.Close() }()

	cfg := search.DefaultConfig()
	cfg.Model = filepath.Base(resolvedModel)
	cfg.OutputDim = cf.dim
	cfg.Overflow = strategy
	cfg.VisionTokensPerImage = cf.visionTokens
	cfg.Normalize = new(cf.normalize)
	cfg.QueryPrefix = *queryPrefix
	cfg.DocumentPrefix = *docPrefix
	cfg.EnableVision = caps.SupportsVision()
	cfg.EnableAudio = caps.SupportsAudio()

	var idx *search.Index
	if *dir != "" {
		idx = search.New(client, cfg)
		stats, idxErr := idx.IndexDir(ctx, *dir)
		if idxErr != nil {
			return 1, idxErr
		}
		_, _ = fmt.Fprintf(out, "Indexed %d item(s) (%d text, %d image, %d audio; skipped %d) in %s [dim=%d]\n",
			stats.Indexed, stats.TextCount, stats.ImageCount, stats.AudioCount, stats.Skipped,
			stats.Duration.Round(time.Millisecond), idx.Dim())
		if *indexPath != "" {
			if saveErr := idx.Save(*indexPath); saveErr != nil {
				return 1, saveErr
			}
			_, _ = fmt.Fprintf(out, "Saved index to %s\n", *indexPath)
		}
	} else {
		var loadErr error
		idx, loadErr = search.Load(*indexPath, client, cfg)
		if loadErr != nil {
			return 1, loadErr
		}
		_, _ = fmt.Fprintf(out, "Loaded %d item(s) from %s [dim=%d]\n", idx.Len(), *indexPath, idx.Dim())
	}

	switch {
	case *queryText != "":
		return executeSearchQuery(out, "text", *queryText, func() ([]search.Match, error) {
			return idx.QueryText(ctx, *queryText, *topK)
		})
	case *queryImage != "":
		return executeSearchQuery(out, "image", *queryImage, func() ([]search.Match, error) {
			return idx.QueryImage(ctx, *queryImage, *topK)
		})
	case *queryAudio != "":
		return executeSearchQuery(out, "audio", *queryAudio, func() ([]search.Match, error) {
			return idx.QueryAudio(ctx, *queryAudio, *topK)
		})
	default:
		if err := runSearchREPL(ctx, idx, *topK, in, out); err != nil {
			return 1, err
		}
		return 0, nil
	}
}

func validateSearchArgs(dir, indexPath, queryText, queryImage, queryAudio string) error {
	if dir == "" && indexPath == "" {
		return fmt.Errorf("-dir or -index is required")
	}
	queryCount := 0
	for _, q := range []string{queryText, queryImage, queryAudio} {
		if strings.TrimSpace(q) != "" {
			queryCount++
		}
	}
	if queryCount > 1 {
		return fmt.Errorf("-query, -query-image, and -query-audio are mutually exclusive")
	}
	return nil
}

func executeSearchQuery(out io.Writer, kind, label string, runFn func() ([]search.Match, error)) (int, error) {
	start := time.Now()
	matches, err := runFn()
	if err != nil {
		return 1, err
	}
	printMatches(out, kind, label, matches, time.Since(start))
	return 0, nil
}

func runSearchREPL(ctx context.Context, idx *search.Index, topK int, in io.Reader, out io.Writer) error {
	_, _ = fmt.Fprintln(out, "Interactive mode. Enter a text query, or :image <path>, :audio <path>, :stats, :help, :quit")
	scanner := bufio.NewScanner(in)
	for {
		if ctx.Err() != nil {
			return nil
		}
		_, _ = fmt.Fprint(out, "\nsearch> ")
		if !scanner.Scan() {
			_, _ = fmt.Fprintln(out)
			return scanner.Err()
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		if strings.HasPrefix(line, ":") {
			cmd, arg, _ := strings.Cut(line, " ")
			arg = unquotePath(arg)
			switch cmd {
			case ":quit", ":exit":
				_, _ = fmt.Fprintln(out, "bye")
				return nil
			case ":help":
				_, _ = fmt.Fprintln(out, "Commands: <text query> | :image <path> | :audio <path> | :stats | :help | :quit")
			case ":stats":
				printIndexStats(out, idx)
			case ":image":
				if arg == "" {
					_, _ = fmt.Fprintln(out, "usage: :image <path>")
					continue
				}
				start := time.Now()
				matches, err := idx.QueryImage(ctx, arg, topK)
				if err != nil {
					_, _ = fmt.Fprintf(out, "error: %v\n", err)
					continue
				}
				printMatches(out, "image", arg, matches, time.Since(start))
			case ":audio":
				if arg == "" {
					_, _ = fmt.Fprintln(out, "usage: :audio <path>")
					continue
				}
				start := time.Now()
				matches, err := idx.QueryAudio(ctx, arg, topK)
				if err != nil {
					_, _ = fmt.Fprintf(out, "error: %v\n", err)
					continue
				}
				printMatches(out, "audio", arg, matches, time.Since(start))
			default:
				_, _ = fmt.Fprintf(out, "unknown command %q (type :help)\n", cmd)
			}
			continue
		}

		start := time.Now()
		matches, err := idx.QueryText(ctx, line, topK)
		if err != nil {
			_, _ = fmt.Fprintf(out, "error: %v\n", err)
			continue
		}
		printMatches(out, "text", line, matches, time.Since(start))
	}
}

func runClassifyCmd(ctx context.Context, args []string, out, errOut io.Writer) (int, error) {
	fs := flag.NewFlagSet("classify", flag.ContinueOnError)
	fs.SetOutput(errOut)
	cf := bindCommonFlags(fs)
	labelsCSV := fs.String("labels", "", "comma-separated candidate labels")
	var labelFlags stringListFlag
	fs.Var(&labelFlags, "label", "candidate label (may be repeated for labels containing spaces or commas)")
	var textInputs stringListFlag
	fs.Var(&textInputs, "text", "text input to classify (may be repeated alongside file arguments)")
	topK := fs.Int("top", 0, "maximum labels to show per input (0 = all)")
	temperature := fs.Float64("temperature", classify.DefaultTemperature, "softmax scaling temperature")
	labelPrefix := fs.String("label-prefix", "", "optional prompt prefix prepended to label strings")
	inputPrefix := fs.String("input-prefix", "", "optional prompt prefix prepended to text inputs")

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0, nil
		}
		return 2, nil
	}

	labels, err := collectLabels(*labelsCSV, labelFlags)
	if err != nil {
		return 2, err
	}
	fileArgs := fs.Args()
	if len(fileArgs) == 0 && len(textInputs) == 0 {
		return 2, fmt.Errorf("provide at least one -text value or input file argument")
	}
	strategy, err := cf.parseOverflow()
	if err != nil {
		return 2, err
	}

	client, _, _, err := cf.openClient(ctx)
	if err != nil {
		return 1, err
	}
	defer func() { _ = client.Close() }()

	cfg := classify.DefaultConfig()
	cfg.OutputDim = cf.dim
	cfg.Overflow = strategy
	cfg.VisionTokensPerImage = cf.visionTokens
	cfg.Normalize = new(cf.normalize)
	cfg.Temperature = *temperature
	cfg.LabelPrefix = *labelPrefix
	cfg.InputPrefix = *inputPrefix

	start := time.Now()
	clf, err := classify.New(ctx, client, labels, cfg)
	if err != nil {
		return 1, err
	}
	_, _ = fmt.Fprintf(out, "Prepared %d label(s) [dim=%d, temp=%.3f] in %s\n",
		len(labels), clf.Dim(), cfg.Temperature, time.Since(start).Round(time.Millisecond))

	if len(textInputs) > 0 {
		qStart := time.Now()
		results, err := clf.ClassifyTexts(ctx, textInputs, *topK)
		if err != nil {
			return 1, err
		}
		perDuration := time.Since(qStart) / time.Duration(len(results))
		for _, res := range results {
			printClassifyResult(out, res, perDuration)
		}
	}

	for _, filePath := range fileArgs {
		qStart := time.Now()
		res, err := clf.ClassifyFile(ctx, filePath, *topK)
		if err != nil {
			return 1, err
		}
		printClassifyResult(out, res, time.Since(qStart))
	}
	return 0, nil
}

type stringListFlag []string

func (s *stringListFlag) String() string {
	return strings.Join(*s, ", ")
}

func (s *stringListFlag) Set(val string) error {
	*s = append(*s, val)
	return nil
}

func collectLabels(csv string, repeated []string) ([]classify.Label, error) {
	var out []classify.Label
	seen := make(map[string]struct{})
	add := func(raw string) {
		name := strings.TrimSpace(raw)
		if name == "" {
			return
		}
		if _, exists := seen[name]; exists {
			return
		}
		seen[name] = struct{}{}
		out = append(out, classify.Label{Name: name})
	}

	if csv != "" {
		for part := range strings.SplitSeq(csv, ",") {
			add(part)
		}
	}
	for _, r := range repeated {
		add(r)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("at least one label is required (-labels or -label)")
	}
	return out, nil
}

func parseOverflow(s string) (litertlm.OverflowStrategy, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "chunk", "chunk_and_average":
		return litertlm.OverflowChunkAndAverage, nil
	case "truncate":
		return litertlm.OverflowTruncate, nil
	case "error":
		return litertlm.OverflowError, nil
	default:
		return 0, fmt.Errorf("unknown overflow strategy %q (use chunk | truncate | error)", s)
	}
}

func printMatches(out io.Writer, kind, query string, matches []search.Match, elapsed time.Duration) {
	_, _ = fmt.Fprintf(out, "\nResults for %s query %q (%s):\n", kind, query, elapsed.Round(time.Millisecond))
	if len(matches) == 0 {
		_, _ = fmt.Fprintln(out, "  (no matches — index is empty)")
		return
	}
	for i, m := range matches {
		_, _ = fmt.Fprintf(out, "  %d. %s  %.4f  [%-5s]  %s",
			i+1, renderBar(m.Score, 14), m.Score, m.Entry.Modality, m.Entry.ID)
		if m.Entry.Modality == search.ModalityText && m.Entry.Preview != "" {
			_, _ = fmt.Fprintf(out, "  -- %s", m.Entry.Preview)
		}
		_, _ = fmt.Fprintln(out)
	}
}

func printIndexStats(out io.Writer, idx *search.Index) {
	var textN, imgN, audN int
	for _, e := range idx.Entries() {
		switch e.Modality {
		case search.ModalityText:
			textN++
		case search.ModalityImage:
			imgN++
		case search.ModalityAudio:
			audN++
		}
	}
	_, _ = fmt.Fprintf(out, "Index stats: model=%s entries=%d dim=%d (text=%d image=%d audio=%d)\n",
		idx.Model(), idx.Len(), idx.Dim(), textN, imgN, audN)
}

func printClassifyResult(out io.Writer, res classify.Result, elapsed time.Duration) {
	_, _ = fmt.Fprintf(out, "\nInput: %s -> Top: %s (sim=%.4f, prob=%.1f%%) [%s]\n",
		res.Input, res.Top.Label, res.Top.Similarity, res.Top.Probability*100, elapsed.Round(time.Millisecond))
	for i, s := range res.Scores {
		_, _ = fmt.Fprintf(out, "  %d. %s  %5.1f%%  sim=%+.4f  %s\n",
			i+1, renderBar(s.Probability, 14), s.Probability*100, s.Similarity, s.Label)
	}
}

func renderBar(value float64, width int) string {
	if width <= 0 {
		return ""
	}
	clamped := min(max(value, 0), 1)
	filled := int(math.Round(clamped * float64(width)))
	return "[" + strings.Repeat("#", filled) + strings.Repeat(".", width-filled) + "]"
}

func unquotePath(s string) string {
	s = strings.TrimSpace(s)
	if len(s) >= 2 && ((s[0] == '"' && s[len(s)-1] == '"') || (s[0] == '\'' && s[len(s)-1] == '\'')) {
		return s[1 : len(s)-1]
	}
	return s
}
