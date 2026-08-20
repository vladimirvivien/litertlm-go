// tool-policy contrasts the two values of WithToolPolicy by running a
// tool whose handler errors for inputs not in its lookup table.
// Selecting -policy=return surfaces the handler error as a Go error
// from Chat.Send; -policy=inform marshals the error message back to
// the model as the tool's result so it can react.
//
// See README.md in this directory for prerequisites and usage.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

const systemPrompt = "You answer weather questions using the lookup_weather tool. Never invent values. If the tool reports an error, acknowledge it and suggest the user try a different city."

// WeatherIn is the model's tool-call argument shape.
type WeatherIn struct {
	City string `description:"city name to look up weather for"`
}

// WeatherOut is one row of the lookup table.
type WeatherOut struct {
	City       string  `json:"city"`
	TempC      float64 `json:"temp_c"`
	Conditions string  `json:"conditions"`
}

var weatherDB = map[string]WeatherOut{
	"Paris":  {City: "Paris", TempC: 12, Conditions: "rain"},
	"Tokyo":  {City: "Tokyo", TempC: 22, Conditions: "clear"},
	"London": {City: "London", TempC: 9, Conditions: "fog"},
	"Lagos":  {City: "Lagos", TempC: 30, Conditions: "humid"},
}

func lookupWeather(_ context.Context, in WeatherIn) (WeatherOut, error) {
	fmt.Printf("(invoke)   lookup_weather(city=%q)\n", in.City)
	out, ok := weatherDB[in.City]
	if !ok {
		return WeatherOut{}, fmt.Errorf("no weather data for city %q", in.City)
	}
	return out, nil
}

func main() {
	model := flag.String("model", "", "path to .litertlm model file (required)")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.16.0)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	prompt := flag.String("prompt", "What's the weather in Atlantis? If that lookup fails, try Tokyo.", "user message")
	policyFlag := flag.String("policy", "", "tool-error policy: return | inform (required)")
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

	policy, err := parsePolicy(*policyFlag)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}

	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	client, err := litertlm.New(ctx,
		litertlm.WithLib(resolvedLib),
		litertlm.WithModel(resolvedModel),
		litertlm.WithBackend(*backend),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	tool, err := litertlm.RegisterTool(client, "lookup_weather",
		"Return the current weather conditions for a city.",
		lookupWeather,
		litertlm.WithToolPolicy(policy),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "register tool: %v\n", err)
		os.Exit(1)
	}

	chat, err := client.NewChat(ctx,
		litertlm.WithSystemPrompt(systemPrompt),
		litertlm.WithTool(tool),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new chat: %v\n", err)
		os.Exit(1)
	}
	defer chat.Close()

	fmt.Printf("policy>    %s\n", *policyFlag)
	fmt.Printf("user>      %s\n", *prompt)
	reply, err := chat.Send(ctx, *prompt)
	if err != nil {
		if hops, ok := errors.AsType[*litertlm.ToolHopsError](err); ok {
			fmt.Fprintf(os.Stderr, "tool hops exceeded after %d iterations; last reply: %s\n",
				hops.Hops, hops.LastReply.Text())
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "send: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("assistant> %s\n", reply.Text())
}

func parsePolicy(s string) (litertlm.ToolPolicy, error) {
	switch s {
	case "return":
		return litertlm.ToolPolicyReturnOnError, nil
	case "inform":
		return litertlm.ToolPolicyInformOnError, nil
	case "":
		return 0, errors.New("--policy is required (return | inform)")
	default:
		return 0, fmt.Errorf("--policy must be return or inform, got %q", s)
	}
}
