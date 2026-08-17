// config-file demonstrates how to configure a LiteRT-LM Client using settings
// loaded from a centralized config.json file via WithConfigFile.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	model := flag.String("model", "", "path to .litertlm model file")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.16.0)")
	configPath := flag.String("config", "config.json", "path to config.json file")
	modelID := flag.String("profile", "gemma", "model ID profile section inside config.json")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries")
	prompt := flag.String("prompt", "What are the three laws of motion?", "prompt text")
	flag.Parse()

	ctx := context.Background()

	resolvedLib := *libPath
	if *getLib != "" {
		staged, err := litertlm.FetchLib("", "", *getLib)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fetch library: %v\n", err)
			os.Exit(1)
		}
		resolvedLib = staged
	}

	resolvedModel := *model
	if resolvedModel == "" {
		resolvedModel = os.Getenv("LITERTLM_MODEL")
	}
	if *getModel != "" {
		staged, err := litertlm.FetchModel(ctx, *getModel)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fetch model: %v\n", err)
			os.Exit(1)
		}
		resolvedModel = staged
	}

	if resolvedModel == "" {
		fmt.Fprintln(os.Stderr, "--model or --get-model (or LITERTLM_MODEL env) is required")
		os.Exit(2)
	}

	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	// Apply WithConfigFile along with model and library paths.
	// Explicit flags or subsequent options can override values in config.json.
	client, err := litertlm.New(ctx,
		litertlm.WithLib(resolvedLib),
		litertlm.WithModel(resolvedModel),
		litertlm.WithConfigFile(*configPath, *modelID),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to initialize client: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	chat, err := client.NewChat(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create chat: %v\n", err)
		os.Exit(1)
	}
	defer chat.Close()

	fmt.Printf("user> %s\n", *prompt)
	reply, err := chat.Send(ctx, *prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "send error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("bot>  %s\n", reply.Text())
}
