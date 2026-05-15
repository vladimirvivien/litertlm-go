// vision asks the model to describe an image, then asks the model
// to briefly assess how well its description aligns with a reference
// description loaded from disk.
//
// Run with a multimodal .litertlm model (e.g. a Gemma 4 multimodal
// variant). See README.md in this directory for invocation details.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	model := flag.String("model", "", "path to a multimodal .litertlm model file (required)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	visionBackend := flag.String("vision-backend", "cpu", "vision backend (cpu | gpu)")
	imagePath := flag.String("image", "", "path to the image file (required)")
	descriptionPath := flag.String("description", "", "path to the reference description file (required)")
	prompt := flag.String("prompt", "Describe this image in 2-3 sentences.", "instruction sent with the image")
	maxTokens := flag.Int("max-tokens", 4096, "engine token budget; vision needs >=4096 because image patches expand into many tokens")
	flag.Parse()

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

	ctx := context.Background()
	litertlm.SetMinLogLevel(litertlm.LogQuiet)
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

	fmt.Printf("=== Image: %s ===\n", *imagePath)

	description, err := client.GenerateMulti(ctx, []litertlm.Part{
		img,
		litertlm.Text(*prompt),
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "describe: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("\n=== Model description ===")
	fmt.Println(strings.TrimSpace(description))

	fmt.Println("\n=== Reference description ===")
	fmt.Println(strings.TrimSpace(string(truth)))

	alignmentPrompt := fmt.Sprintf(
		"Reference description:\n\n%s\n\nYour description:\n\n%s\n\n"+
			"Assess in 2-3 sentences how well your description aligns with the reference. "+
			"State whether they describe the same scene, the main points of agreement, "+
			"and any clear contradictions.",
		strings.TrimSpace(string(truth)),
		strings.TrimSpace(description),
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
