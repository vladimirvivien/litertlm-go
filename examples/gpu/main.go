// gpu demonstrates GPU-accelerated local inference plus BenchmarkInfo
// readout (init time, time-to-first-token, prefill/decode throughput).
//
// See README.md in this directory for the GPU-specific build, the extra
// runtime libraries that must be staged in LITERTLM_LIB, and instructions
// for getting the wrapper to load the GPU-capable build by name.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	model := flag.String("model", "", "path to .litertlm model file")
	prompt := flag.String("prompt", "Summarise Go's approach to concurrency in one paragraph.", "prompt text")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the GPU-capable LiteRT-LM shared library + GPU plugin .so/.dylib/.dll files")
	maxTokens := flag.Int("max", 1024, "max total tokens (prompt + output); must be >= the model's smallest prefill signature, typically 128")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "--model is required")
		os.Exit(2)
	}

	if err := litertlm.Load(*libPath, "gpu"); err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer litertlm.Close()

	// Silence LiteRT-LM's INFO/WARN chatter. Bump to LogInfo (0) to see it.
	litertlm.SetMinLogLevel(litertlm.LogError)

	settings, err := litertlm.NewEngineSettings(*model, "gpu", nil, nil)
	if err != nil {
		fmt.Fprintf(os.Stderr, "settings: %v\n", err)
		os.Exit(1)
	}
	defer settings.Delete()
	settings.SetMaxNumTokens(*maxTokens)
	settings.EnableBenchmark()

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

	b, err := session.BenchmarkInfo()
	if err != nil {
		fmt.Fprintf(os.Stderr, "benchmark: %v\n", err)
		return
	}
	defer b.Delete()

	fmt.Println("--- GPU benchmark ---")
	fmt.Printf("Total init time:       %.3f s\n", b.TotalInitTime())
	fmt.Printf("Time to first token:   %.3f s\n", b.TimeToFirstToken())
	for i := 0; i < b.NumPrefillTurns(); i++ {
		fmt.Printf("Prefill turn %d:        %.1f tokens/sec (%d tokens)\n",
			i, b.PrefillTokensPerSec(i), b.PrefillTokenCount(i))
	}
	for i := 0; i < b.NumDecodeTurns(); i++ {
		fmt.Printf("Decode  turn %d:        %.1f tokens/sec (%d tokens)\n",
			i, b.DecodeTokensPerSec(i), b.DecodeTokenCount(i))
	}
}
