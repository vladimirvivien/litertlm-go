// token-scores scores one or more candidate continuations against
// the same prefilled prompt and prints the per-token scores
// alongside the detokenized text. One Session per candidate — the
// CPU engine rejects num_targets > 1 in a single ScoreTexts call.
//
// See README.md in this directory for prerequisites and usage.
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	model := flag.String("model", "", "path to .litertlm model file (required)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	prompt := flag.String("prompt", "The capital of France is", "prefilled prompt")
	candidates := flag.String("candidates", " Paris, London", "comma-separated continuations to score (each becomes its own Session)")
	maxTokens := flag.Int("max-tokens", 2048, "engine total token budget")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "--model is required")
		os.Exit(2)
	}

	if err := litertlm.Load(*libPath, *backend, ""); err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer litertlm.Close()
	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	settings, err := litertlm.NewEngineSettings(*model, *backend, nil, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "settings: %v\n", err)
		os.Exit(1)
	}
	defer settings.Delete()
	settings.SetMaxNumTokens(*maxTokens)

	engine, err := litertlm.NewEngine(settings)
	if err != nil {
		fmt.Fprintf(os.Stderr, "engine: %v\n", err)
		os.Exit(1)
	}
	defer engine.Delete()

	fmt.Printf("prompt: %q\n\n", *prompt)

	for raw := range strings.SplitSeq(*candidates, ",") {
		text := strings.TrimSpace(raw)
		if text == "" {
			continue
		}
		// SentencePiece encodes a leading space as the ▁ marker;
		// the score depends on it.
		candidate := " " + text
		if err := scoreOne(engine, *prompt, candidate); err != nil {
			fmt.Fprintf(os.Stderr, "score %q: %v\n", candidate, err)
			os.Exit(1)
		}
	}
}

func scoreOne(engine litertlm.Engine, prompt, candidate string) error {
	session, err := engine.NewSession(0)
	if err != nil {
		return fmt.Errorf("session: %w", err)
	}
	defer session.Delete()

	if err = session.RunPrefill([]litertlm.InputData{
		litertlm.NewTextInputString(prompt),
	}); err != nil {
		return fmt.Errorf("prefill: %w", err)
	}

	resp, err := session.ScoreTexts([]string{candidate}, true)
	if err != nil {
		return fmt.Errorf("score: %w", err)
	}
	defer resp.Delete()

	total, _ := resp.Score(0)
	tokenScores, hasTokens := resp.TokenScores(0)

	ids, err := engine.Tokenize(candidate)
	if err != nil {
		return fmt.Errorf("tokenize: %w", err)
	}

	fmt.Printf("candidate:   %q\n", candidate)
	fmt.Printf("total score: %.4f\n", total)
	if !hasTokens || len(tokenScores) == 0 {
		fmt.Println("(no per-token scores)")
		fmt.Println()
		return nil
	}

	fmt.Printf("  %-10s %-12s %s\n", "token_id", "token", "score")
	for i, id := range ids {
		tokText, derr := engine.Detokenize([]int32{id})
		if derr != nil {
			tokText = ""
		}
		var score string
		if i < len(tokenScores) {
			score = fmt.Sprintf("%.4f", tokenScores[i])
		} else {
			score = "-"
		}
		fmt.Printf("  %-10d %-12q %s\n", id, tokText, score)
	}
	fmt.Println()
	return nil
}
