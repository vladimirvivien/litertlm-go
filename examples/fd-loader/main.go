// fd-loader demonstrates how to initialize the LiteRT-LM engine from an open file
// descriptor instead of a filesystem path string.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"runtime"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	if runtime.GOOS == "windows" {
		fmt.Fprintln(os.Stderr, "fd-loader is unsupported on Windows due to CRT DLL boundary descriptor restrictions. Exiting.")
		os.Exit(0)
	}

	model := flag.String("model", "", "path to .litertlm model file")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries")
	prompt := flag.String("prompt", "Paris is the capital of", "prompt text")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "--model is required")
		os.Exit(2)
	}

	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	// Open the model file to obtain a raw system file descriptor
	file, err := os.Open(*model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to open model file: %v\n", err)
		os.Exit(1)
	}
	// Note: We do not close the file here because the engine takes ownership
	// of the file descriptor and will close it when the engine is destroyed.

	ctx := context.Background()
	client, err := litertlm.New(ctx,
		litertlm.WithLib(*libPath),
		litertlm.WithModelFd(int(file.Fd())),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		_ = file.Close()
		os.Exit(1)
	}
	defer client.Close()

	// Simple generation check to verify it loaded correctly.
	res, err := client.Generate(ctx, *prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Generate: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Prompt: %s\n", *prompt)
	fmt.Printf("Response: %s\n", res)
}
