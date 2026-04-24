// stream demonstrates token-by-token streaming generation using a Go channel.
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
	prompt := flag.String("prompt", "Write a short haiku about the sea.", "prompt text")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries (falls back to LITERTLM_LIB env)")
	maxTokens := flag.Int("max", 1024, "max total tokens (prompt + output); must be >= the model's smallest prefill signature, typically 128")
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

	// Silence LiteRT-LM's INFO/WARN chatter. Bump to LogInfo (0) to see it.
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

	inputs := []litertlm.InputData{litertlm.NewTextInputString(*prompt)}
	for chunk := range session.GenerateContentStreamCh(inputs) {
		if chunk.Err != nil {
			fmt.Fprintf(os.Stderr, "\nstream error: %v\n", chunk.Err)
			os.Exit(1)
		}
		fmt.Print(chunk.Text)
		if chunk.Final {
			fmt.Println()
		}
	}
}
