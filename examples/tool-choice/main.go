// tool-choice demonstrates OpenAI-compatible per-turn ToolChoice policies
// under constrained decoding (requires LiteRT-LM C API 1.1.0+).
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

type weatherArgs struct {
	City string `json:"city" jsonschema:"description=City name,required"`
}

func main() {
	model := flag.String("model", "", "path to .litertlm model file (falls back to LITERTLM_MODEL env)")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.19.1)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	flag.Parse()

	ctx := context.Background()
	resolvedLib := *libPath
	if *getLib != "" {
		staged, err := litertlm.FetchLib(ctx, "", "", *getLib)
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
	client, err := litertlm.New(ctx,
		litertlm.WithLib(resolvedLib),
		litertlm.WithModel(resolvedModel),
		litertlm.WithBackend(*backend),
		litertlm.WithMaxTokens(1024),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = client.Close() }()

	weatherTool, err := litertlm.RegisterTool(client, "get_weather", "Return current weather for a city",
		func(_ context.Context, a weatherArgs) (map[string]string, error) {
			return map[string]string{"city": a.City, "condition": "sunny"}, nil
		},
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "register weather tool: %v\n", err)
		os.Exit(1)
	}
	timeTool, err := litertlm.RegisterTool(client, "get_time", "Return current local time for a city",
		func(_ context.Context, a weatherArgs) (map[string]string, error) {
			return map[string]string{"city": a.City, "time": "10:30 AM"}, nil
		},
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "register time tool: %v\n", err)
		os.Exit(1)
	}

	chat, err := client.NewChat(ctx,
		litertlm.WithConstrainedDecoding(true),
		litertlm.WithTool(weatherTool, timeTool),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new chat: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = chat.Close() }()

	cases := []struct {
		label  string
		prompt string
		choice litertlm.ToolChoice
	}{
		{"forced function (get_weather)", "How is Boston right now?", litertlm.ToolChoiceFunction("get_weather")},
		{"allowed_tools (get_time only)", "Check Boston details.", litertlm.ToolChoiceAllowedTools("required", "get_time")},
		{"none (text reply only)", "Say hello without calling tools.", litertlm.ToolChoiceNone()},
	}

	for _, tc := range cases {
		reply, sendErr := chat.Send(ctx, tc.prompt,
			litertlm.WithMaxOutputTokens(64),
			litertlm.WithToolChoice(tc.choice),
		)
		if errors.Is(sendErr, litertlm.ErrUnimplemented) {
			maj, minVer, pat, _ := litertlm.APIVersion()
			fmt.Printf("WithToolChoice requires LiteRT-LM C API 1.1.0+ (loaded %d.%d.%d): %v\n",
				maj, minVer, pat, sendErr)
			return
		}
		if sendErr != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", tc.label, sendErr)
			os.Exit(1)
		}
		fmt.Printf("[%s] %s\n", tc.label, reply.Text())
	}
}
