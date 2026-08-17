// extract uses litertlm.GenerateDataMulti[T] to pull a typed Scene
// struct out of an image, prints the JSON, then asks the model to
// briefly assess how well the extracted JSON aligns with a reference
// description loaded from disk.
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
	"strings"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

// Scene is a generic image-extraction target: a one-line description
// plus a list of distinct objects.
type Scene struct {
	Description string   `json:"description"`
	Objects     []string `json:"objects"`
}

func main() {
	model := flag.String("model", "", "path to a multimodal .litertlm model file (required)")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.16.0)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	visionBackend := flag.String("vision-backend", "cpu", "vision backend (cpu | gpu)")
	imagePath := flag.String("image", "", "path to the image file (required)")
	descriptionPath := flag.String("description", "", "path to the reference description file (required)")
	prompt := flag.String("prompt", "Extract a one-sentence scene description and a list of distinct objects visible in the image.", "instruction sent with the image")
	retries := flag.Int("retries", 2, "max retry attempts on parse failure")
	maxTokens := flag.Int("max-tokens", 4096, "engine token budget; vision needs >=4096 because image patches expand into many tokens")
	flag.Parse()

	ctx := context.Background()

	resolvedLib := *libPath
	if *getLib != "" {
		staged, err := litertlm.FetchLib("", "", *getLib)
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

	for _, req := range []struct {
		name, val string
	}{
		{"--model", *model},
		{"--image", *imagePath},
		{"--description", *descriptionPath},
	} {
		if req.val == "" {
			fmt.Fprintf(os.Stderr, "%s is required\n", req.name)
			os.Exit(2)
		}
	}

	truth, err := os.ReadFile(*descriptionPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read %s: %v\n", *descriptionPath, err)
		os.Exit(1)
	}

	img, err := litertlm.ImageFromFile(*imagePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load image: %v\n", err)
		os.Exit(1)
	}

	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	client, err := litertlm.New(ctx,
		litertlm.WithLib(resolvedLib),
		litertlm.WithModel(resolvedModel),
		litertlm.WithBackend(*backend),
		litertlm.WithVisionBackend(*visionBackend),
		litertlm.WithMaxTokens(*maxTokens),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	fmt.Printf("=== Image: %s ===\n", *imagePath)

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

	fmt.Println("\n=== Reference description ===")
	fmt.Println(strings.TrimSpace(string(truth)))

	alignmentPrompt := fmt.Sprintf(
		"Reference description:\n\n%s\n\nExtracted JSON:\n\n%s\n\n"+
			"Assess in 2-3 sentences how well the extracted JSON aligns with the reference. "+
			"Name the elements that match, anything in the reference that the JSON missed, "+
			"and anything in the JSON not present in the reference.",
		strings.TrimSpace(string(truth)),
		string(pretty),
	)
	chat, err := client.NewChat(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new chat: %v\n", err)
		os.Exit(1)
	}
	defer chat.Close()
	reply, err := chat.Send(ctx, alignmentPrompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "alignment: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("\n=== Alignment assessment ===")
	fmt.Println(strings.TrimSpace(reply.Text()))
}
