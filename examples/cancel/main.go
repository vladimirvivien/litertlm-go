// cancel demonstrates aborting an in-flight streaming generation.
// We start a long-running generation, read the first N chunks, then
// call Session.Cancel() to ask the engine to stop. The stream channel
// closes shortly after with the in-progress final chunk.
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
	prompt := flag.String("prompt", "Tell me a long story about a dragon and a wizard.", "prompt text")
	cancelAfter := flag.Int("cancel-after", 8, "number of chunks to receive before cancelling")
	maxTokens := flag.Int("max", 4096, "max total tokens (prompt + output)")
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

	fmt.Printf("prompt:  %s\n", *prompt)
	fmt.Println("output:")

	start := time.Now()
	stream := session.GenerateContentStreamCh([]litertlm.InputData{
		litertlm.NewTextInputString(*prompt),
	})

	count := 0
	cancelled := false
	for chunk := range stream {
		if chunk.Err != nil {
			fmt.Fprintf(os.Stderr, "\nstream error: %v\n", chunk.Err)
			break
		}
		fmt.Print(chunk.Text)
		count++
		if !cancelled && count >= *cancelAfter && !chunk.Final {
			cancelled = true
			fmt.Printf("\n\n[%v] cancelling after %d chunks ...\n",
				time.Since(start).Round(time.Millisecond), count)
			session.Cancel()
		}
		if chunk.Final {
			fmt.Println()
		}
	}
	fmt.Printf("done (%d chunks total, wall=%v)\n",
		count, time.Since(start).Round(time.Millisecond))
}
