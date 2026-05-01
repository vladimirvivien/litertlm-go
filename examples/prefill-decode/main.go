// prefill-decode demonstrates the explicit two-phase generation flow:
// RunPrefill seeds the session with the prompt context, RunDecode then
// produces the response. This is the lower-level alternative to the
// all-in-one Session.GenerateContent.
//
// See README.md in this directory for prerequisites and usage.
package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	model := flag.String("model", "", "path to .litertlm model file")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries (falls back to LITERTLM_LIB env)")
	prompt := flag.String("prompt", "The capital of France is", "prompt text")
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

	// ---- Phase 1: prefill ----
	prefillStart := time.Now()
	if perr := session.RunPrefill([]litertlm.InputData{
		litertlm.NewTextInputString(*prompt),
	}); perr != nil {
		fmt.Fprintf(os.Stderr, "prefill: %v\n", perr)
		os.Exit(1)
	}
	fmt.Printf("prefill:  %v   (%q)\n", time.Since(prefillStart).Round(time.Millisecond), *prompt)

	// ---- Phase 2: decode ----
	decodeStart := time.Now()
	resp, err := session.RunDecode()
	if err != nil {
		fmt.Fprintf(os.Stderr, "decode: %v\n", err)
		os.Exit(1)
	}
	defer resp.Delete()
	fmt.Printf("decode:   %v\n", time.Since(decodeStart).Round(time.Millisecond))
	fmt.Printf("response: %s\n", resp.Text(0))
}
