// extract demonstrates structured extraction from an image using
// litertlm.GenerateDataMulti[T].
//
// The example:
//  1. Loads an image and a Markdown ground-truth sidecar from
//     ../testdata/ (defaults to img2.png + img2.md).
//  2. Calls GenerateDataMulti[Scene] for typed JSON extraction.
//  3. Asks the model to compare its extracted JSON to the ground
//     truth and call out agreements and differences.
//
// Run with a multimodal .litertlm model (e.g. a Gemma 4 multimodal
// variant). See README.md in this directory for invocation details.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

// Scene is a generic image-extraction target: a one-line description
// plus a list of distinct objects. The shape is deliberately open —
// vision-language models are typically more reliable producing this
// kind of soft inventory than a domain-specific schema.
type Scene struct {
	Description string   `json:"description"`
	Objects     []string `json:"objects"`
}

func main() {
	model := flag.String("model", "", "path to a multimodal .litertlm model file (required)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	visionBackend := flag.String("vision-backend", "cpu", "vision backend (cpu | gpu)")
	testdata := flag.String("testdata", "../testdata", "directory holding the image and .md sidecar")
	name := flag.String("name", "img2", "basename of the image and matching .md file in --testdata")
	prompt := flag.String("prompt", "Extract a one-sentence scene description and a list of distinct objects visible in the image.", "instruction sent with the image")
	retries := flag.Int("retries", 2, "max retry attempts on parse failure")
	maxTokens := flag.Int("max-tokens", 4096, "engine token budget (vision needs >=4096 — image patches expand into many tokens)")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "--model is required")
		os.Exit(2)
	}

	imagePath, err := findImage(*testdata, *name)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	mdPath := filepath.Join(*testdata, *name+".md")
	truth, err := os.ReadFile(mdPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read %s: %v\n", mdPath, err)
		os.Exit(1)
	}

	img, err := litertlm.ImageFromFile(imagePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load image: %v\n", err)
		os.Exit(1)
	}

	ctx := context.Background()
	client, err := litertlm.New(ctx,
		litertlm.WithLib(*libPath),
		litertlm.WithModel(*model),
		litertlm.WithBackend(*backend),
		litertlm.WithVisionBackend(*visionBackend),
		litertlm.WithMaxTokens(*maxTokens),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	fmt.Printf("=== Image: %s ===\n", imagePath)

	// Step 1: structured extraction via GenerateDataMulti[Scene].
	scene, err := litertlm.GenerateDataMulti[Scene](ctx, client,
		[]litertlm.Part{img, litertlm.Text(*prompt)},
		litertlm.WithRetries(*retries),
	)
	if err != nil {
		var gd *litertlm.GenerateDataError
		if errors.As(err, &gd) && gd.Phase == "parse" {
			fmt.Fprintf(os.Stderr,
				"parse failed after %d attempt(s); raw model output:\n%s\n",
				gd.Attempts, gd.Raw)
		} else {
			fmt.Fprintf(os.Stderr, "extract: %v\n", err)
		}
		os.Exit(1)
	}
	pretty, _ := json.MarshalIndent(scene, "", "  ")
	fmt.Println("\n=== Extracted Scene ===")
	fmt.Println(string(pretty))

	// Step 2: text-only call asking the model to compare its
	// extracted JSON to the ground truth in img.md.
	comparePrompt := fmt.Sprintf(
		"You extracted the following scene from an image:\n\n%s\n\n"+
			"The reference description is:\n\n%s\n\n"+
			"Compare your extraction to the reference. List the elements you correctly identified, any you missed, and any you added.",
		string(pretty),
		strings.TrimSpace(string(truth)),
	)
	comparison, err := client.Generate(ctx, comparePrompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "compare: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("\n=== Reference (ground truth) ===")
	fmt.Println(strings.TrimSpace(string(truth)))
	fmt.Println("\n=== Model self-comparison ===")
	fmt.Println(strings.TrimSpace(comparison))
}

// findImage looks for <name>.<ext> under dir, trying the extensions
// litertlm.ImageFromFile knows how to MIME-tag.
func findImage(dir, name string) (string, error) {
	exts := []string{".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
	tried := make([]string, 0, len(exts))
	for _, ext := range exts {
		path := filepath.Join(dir, name+ext)
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
		tried = append(tried, path)
	}
	return "", fmt.Errorf("image not found; tried: %s", strings.Join(tried, ", "))
}
