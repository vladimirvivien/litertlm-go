// embed demonstrates standalone text, batch, and optional multimodal embedding
// generation using EmbeddingClient, OverflowStrategy, normalization, and
// Matryoshka output dimension truncation.
package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	model := flag.String("model", "", "path to an embedding .litertlm model file (or set LITERTLM_MODEL)")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.19.1)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu | npu)")
	dim := flag.Int("dim", 0, "optional Matryoshka truncated output dimension (0 = full model dimension)")
	normalize := flag.Bool("normalize", true, "L2-normalize output embeddings")
	overflow := flag.String("overflow", "chunk", "overflow strategy (chunk | truncate | error)")
	imagePath := flag.String("image", "", "optional path to an image file to embed alongside query A")
	textA := flag.String("a", "Go concurrency with goroutines and channels", "first text input")
	textB := flag.String("b", "Lightweight concurrent execution in Go", "semantically similar text input")
	textC := flag.String("c", "Freshly baked sourdough bread recipe", "unrelated comparison text input")
	flag.Parse()

	ctx := context.Background()
	resolvedLib := *libPath
	if *getLib != "" {
		staged, err := litertlm.FetchLib(ctx, "", "", *getLib)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fetch library: %v\n", err)
			os.Exit(1)
		}
		resolvedLib = staged
	}

	resolvedModel := *model
	if resolvedModel == "" {
		resolvedModel = os.Getenv("LITERTLM_MODEL")
	}
	if *getModel != "" {
		staged, err := litertlm.FetchModel(ctx, *getModel)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fetch model: %v\n", err)
			os.Exit(1)
		}
		resolvedModel = staged
	}
	if resolvedModel == "" {
		fmt.Fprintln(os.Stderr, "--model or --get-model (or LITERTLM_MODEL env) is required")
		os.Exit(2)
	}

	if resolvedLib != "" {
		if err := litertlm.Load(resolvedLib, *backend, ""); err != nil {
			fmt.Fprintf(os.Stderr, "load library: %v\n", err)
			os.Exit(1)
		}
	}

	caps, err := litertlm.InspectModel(resolvedModel)
	if err != nil {
		fmt.Fprintf(os.Stderr, "inspect model: %v\n", err)
		os.Exit(1)
	}
	if caps.ModelType != litertlm.ModelTypeEmbedding {
		fmt.Fprintf(os.Stderr, "model %s has ModelType=%s; an Embedding .litertlm model is required\n",
			resolvedModel, caps.ModelType)
		os.Exit(1)
	}

	strategy, err := parseOverflow(*overflow)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(2)
	}

	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	client, err := litertlm.NewEmbeddingClient(ctx,
		litertlm.WithLib(resolvedLib),
		litertlm.WithModel(resolvedModel),
		litertlm.WithBackend(*backend),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new embedding client: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = client.Close() }()

	embedOpts := []litertlm.EmbeddingOption{
		litertlm.WithNormalize(*normalize),
		litertlm.WithOverflowStrategy(strategy),
	}
	if *dim > 0 {
		embedOpts = append(embedOpts, litertlm.WithOutputSize(*dim))
	}

	vecs, err := client.EmbedBatch(ctx, []string{*textA, *textB, *textC}, embedOpts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "embed batch: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Model: %s (native_dim=%d, output_dim=%d, normalize=%v, overflow=%s)\n",
		resolvedModel, caps.EmbeddingDimension, len(vecs[0]), *normalize, strategy)
	fmt.Printf("cos(A, B) [related]:   %.4f\n", cosineSimilarity(vecs[0], vecs[1]))
	fmt.Printf("cos(A, C) [unrelated]: %.4f\n", cosineSimilarity(vecs[0], vecs[2]))

	if *imagePath != "" {
		img, err := litertlm.ImageFromFile(*imagePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "read image: %v\n", err)
			os.Exit(1)
		}
		multiVec, err := client.EmbedMulti(ctx, []litertlm.Part{img, litertlm.Text(*textA)}, embedOpts...)
		if err != nil {
			fmt.Fprintf(os.Stderr, "embed multi: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Image+text embedding dimension: %d (cos with A: %.4f)\n",
			len(multiVec), cosineSimilarity(multiVec, vecs[0]))
	}
}

func parseOverflow(s string) (litertlm.OverflowStrategy, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "chunk", "chunk_and_average":
		return litertlm.OverflowChunkAndAverage, nil
	case "truncate":
		return litertlm.OverflowTruncate, nil
	case "error":
		return litertlm.OverflowError, nil
	default:
		return 0, fmt.Errorf("unknown overflow strategy %q (use chunk | truncate | error)", s)
	}
}

func cosineSimilarity(a, b []float32) float64 {
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
