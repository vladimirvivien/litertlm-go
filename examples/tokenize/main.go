// tokenize demonstrates the engine's tokenizer round-trip:
// text → []int32 token ids → text. No inference is run, so it's a
// quick way to inspect how a chat-tuned model splits a prompt.
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
	text := flag.String("text", "Hello, world. How are you today?", "text to tokenize")
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

	engine, err := litertlm.NewEngine(settings)
	if err != nil {
		fmt.Fprintf(os.Stderr, "engine: %v\n", err)
		os.Exit(1)
	}
	defer engine.Delete()

	tokens, err := engine.Tokenize(*text)
	if err != nil {
		fmt.Fprintf(os.Stderr, "tokenize: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("text:        %q\n", *text)
	fmt.Printf("tokens (%d):  %v\n", len(tokens), tokens)

	roundTrip, err := engine.Detokenize(tokens)
	if err != nil {
		fmt.Fprintf(os.Stderr, "detokenize: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("round-trip:  %q\n", roundTrip)

	// Model-metadata token sequences.
	start, err := engine.StartTokenIDs()
	if err != nil {
		fmt.Fprintf(os.Stderr, "start tokens: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("start token: %v\n", start)

	stops, err := engine.StopTokenIDs()
	if err != nil {
		fmt.Fprintf(os.Stderr, "stop tokens: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("stop tokens: %v\n", stops)
}
