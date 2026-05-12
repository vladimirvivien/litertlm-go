// audio transcribes or answers a question about a short audio clip
// using a multimodal model. With -transcript set, it then asks the
// model to assess how closely its own transcript matches a reference
// loaded from disk.
//
// Run with a multimodal .litertlm model that includes an audio
// encoder (e.g. a Gemma 4 multimodal variant). See README.md in this
// directory for invocation details.
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
	audioBackend := flag.String("audio-backend", "cpu", "audio backend (cpu | gpu)")
	audioPath := flag.String("audio", "", "path to the audio file: .wav / .mp3 / .ogg / .flac / .m4a / .aac (required)")
	audioMime := flag.String("audio-mime", "", "override MIME type (default: derived from file extension via AudioFromFile)")
	transcriptPath := flag.String("transcript", "", "optional path to a reference transcript; if set, the model assesses alignment with its own output")
	prompt := flag.String("prompt", "Transcribe the speech in this audio clip verbatim.", "instruction sent with the audio")
	maxTokens := flag.Int("max-tokens", 4096, "engine token budget; audio frames expand into many tokens like image patches")
	flag.Parse()

	for _, req := range []struct {
		name, val string
	}{
		{"--model", *model},
		{"--audio", *audioPath},
	} {
		if req.val == "" {
			fmt.Fprintf(os.Stderr, "%s is required\n", req.name)
			os.Exit(2)
		}
	}

	audio, err := loadAudio(*audioPath, *audioMime)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load audio: %v\n", err)
		os.Exit(1)
	}

	ctx := context.Background()
	client, err := litertlm.New(ctx,
		litertlm.WithLib(*libPath),
		litertlm.WithModel(*model),
		litertlm.WithBackend(*backend),
		litertlm.WithAudioBackend(*audioBackend),
		litertlm.WithMaxTokens(*maxTokens),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	fmt.Printf("=== Audio: %s (%s) ===\n", *audioPath, displayMime(audio.Mime()))

	out, err := client.GenerateMulti(ctx, []litertlm.Part{
		audio,
		litertlm.Text(*prompt),
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "generate: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("\n=== Model output ===")
	fmt.Println(strings.TrimSpace(out))

	if *transcriptPath == "" {
		return
	}
	truth, err := os.ReadFile(*transcriptPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read transcript: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("\n=== Reference transcript ===")
	fmt.Println(strings.TrimSpace(string(truth)))

	alignmentPrompt := fmt.Sprintf(
		"Reference transcript:\n\n%s\n\nYour transcript:\n\n%s\n\n"+
			"Assess in 2-3 sentences how closely your transcript matches the reference. "+
			"Call out missing words, substitutions, and any clear errors.",
		strings.TrimSpace(string(truth)),
		strings.TrimSpace(out),
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

// loadAudio uses AudioFromFile when no MIME override is given (MIME
// is inferred from the file extension) and AudioWithMime otherwise.
// The override path is useful when the file extension does not match
// the actual container (e.g. raw PCM saved as .bin).
func loadAudio(path, mimeOverride string) (litertlm.Part, error) {
	if mimeOverride == "" {
		return litertlm.AudioFromFile(path)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return litertlm.Part{}, err
	}
	return litertlm.AudioWithMime(b, mimeOverride), nil
}

func displayMime(m string) string {
	if m == "" {
		return "mime unknown"
	}
	return m
}
