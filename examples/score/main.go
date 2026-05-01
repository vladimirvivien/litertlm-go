// score demonstrates per-target text scoring: prefill the prompt, then
// score one candidate completion and inspect its log-probability score
// and tokenized length.
//
// See README.md in this directory for prerequisites and usage.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	model := flag.String("model", "", "path to .litertlm model file")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries (falls back to LITERTLM_LIB env)")
	prompt := flag.String("prompt", "The capital of France is", "prefilled prompt")
	target := flag.String("target", " Paris.", "candidate completion to score")
	maxTokens := flag.Int("max", 2048, "max total tokens (prompt + output)")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "--model is required")
		os.Exit(2)
	}

	if err := litertlm.Load(*libPath, *backend); err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer litertlm.Close()
	litertlm.SetMinLogLevel(litertlm.LogError)

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

	session, err := engine.NewSession(0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "session: %v\n", err)
		os.Exit(1)
	}
	defer session.Delete()

	// Prefill must run before scoring so the engine has the context
	// against which to score the candidate.
	if perr := session.RunPrefill([]litertlm.InputData{
		litertlm.NewTextInputString(*prompt),
	}); perr != nil {
		fmt.Fprintf(os.Stderr, "prefill: %v\n", perr)
		os.Exit(1)
	}

	// The CPU engine currently rejects num_targets > 1, so the slice
	// holds exactly one candidate.
	resp, err := session.ScoreTexts([]string{*target}, true)
	if err != nil {
		fmt.Fprintf(os.Stderr, "score: %v\n", err)
		os.Exit(1)
	}
	defer resp.Delete()

	fmt.Printf("prompt:    %q\n", *prompt)
	fmt.Printf("candidate: %q\n", *target)
	for i := 0; i < resp.NumCandidates(); i++ {
		s, sok := resp.Score(i)
		t, tok := resp.TokenLength(i)
		fmt.Printf("[%d] text=%q score=%v (ok=%v) tokenLen=%v (ok=%v)\n",
			i, resp.Text(i), s, sok, t, tok)
	}
}
