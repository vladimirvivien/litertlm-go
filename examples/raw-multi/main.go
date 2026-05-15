// raw-multi runs the same image + caption-text input through each
// of Client.GenerateMulti, Client.GenerateMultiStream, and
// Client.GenerateMultiResponse to surface the call-shape and
// return-type differences between the three *Multi siblings.
//
// See README.md in this directory for prerequisites and usage.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	model := flag.String("model", "", "path to .litertlm model file (required)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "LLM inference backend (cpu | gpu)")
	visionBackend := flag.String("vision-backend", "cpu", "vision encoder backend (cpu | gpu)")
	imagePath := flag.String("image", "", "path to image file (required)")
	prompt := flag.String("prompt", "Describe this image in one sentence.", "caption prompt issued for all three calls")
	maxTokens := flag.Int("max-tokens", 128, "per-call decode cap (WithMaxOutputTokens)")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "--model is required")
		os.Exit(2)
	}
	if *imagePath == "" {
		fmt.Fprintln(os.Stderr, "--image is required")
		os.Exit(2)
	}

	img, err := litertlm.ImageFromFile(*imagePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load image: %v\n", err)
		os.Exit(1)
	}

	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	ctx := context.Background()
	client, err := litertlm.New(ctx,
		litertlm.WithLib(*libPath),
		litertlm.WithModel(*model),
		litertlm.WithBackend(*backend),
		litertlm.WithVisionBackend(*visionBackend),
		litertlm.WithBenchmarkEnabled(),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	parts := []litertlm.Part{img, litertlm.Text(*prompt)}
	opts := []litertlm.GenOption{litertlm.WithMaxOutputTokens(*maxTokens)}

	fmt.Printf("image:  %s\n", *imagePath)
	fmt.Printf("prompt: %s\n\n", *prompt)

	fmt.Println("=== GenerateMulti (synchronous) ===")
	text, err := client.GenerateMulti(ctx, parts, opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "GenerateMulti: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(text)

	fmt.Println("\n=== GenerateMultiStream ===")
	chunks := 0
	for chunk, serr := range client.GenerateMultiStream(ctx, parts, opts...) {
		if serr != nil {
			fmt.Fprintf(os.Stderr, "\nstream: %v\n", serr)
			os.Exit(1)
		}
		fmt.Print(chunk.Text)
		chunks++
	}
	fmt.Printf("\n(chunks delivered: %d)\n", chunks)

	fmt.Println("\n=== GenerateMultiResponse ===")
	resp, err := client.GenerateMultiResponse(ctx, parts, opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "GenerateMultiResponse: %v\n", err)
		os.Exit(1)
	}
	replyText := resp.Text()
	fmt.Println(replyText)
	if n, err := client.TokenLength(replyText); err == nil {
		fmt.Printf("token length: %d\n", n)
	}
	if b := resp.Benchmark(); b != nil {
		fmt.Printf("time to first token: %v\n", b.TimeToFirstToken)
		if len(b.PrefillTokensPerSec) > 0 {
			fmt.Printf("prefill: %.1f tok/s\n", b.PrefillTokensPerSec[0])
		}
		if len(b.DecodeTokensPerSec) > 0 {
			fmt.Printf("decode:  %.1f tok/s\n", b.DecodeTokensPerSec[0])
		}
	}
}
