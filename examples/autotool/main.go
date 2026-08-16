// autotool demonstrates typed tool registration and auto-dispatch.
//
// RegisterTool reflects over the input struct to build a JSON-Schema
// parameters map and registers the tool on the Client. Attached to a
// Chat with WithTool, the framework dispatches the handler when the
// model invokes the tool and returns the post-tool natural-language
// answer in a single Chat.Send call — no manual SendToolResult step.
//
// See examples/conversation for the lower-level path that uses
// NewRawTool + Reply.ToolCalls + Chat.SendToolResult instead.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

const systemPrompt = "You are a calculator assistant. When the user asks for an arithmetic result, ALWAYS call the appropriate tool — never compute the answer in your head."

// AddIn is the typed input the model's tool-call arguments are
// unmarshaled into. Field names use the json tag where set, else the
// lowercased Go name. The description tag becomes the field's
// description in the generated JSON Schema.
type AddIn struct {
	A int `description:"first addend"`
	B int `description:"second addend"`
}

// AddOut is the typed result the framework marshals as the tool
// response back to the model.
type AddOut struct {
	Sum int `json:"sum"`
}

func main() {
	model := flag.String("model", "", "path to .litertlm model file")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.16.0)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries (falls back to LITERTLM_LIB env)")
	prompt := flag.String("prompt", "What is 17 plus 25?", "user question")
	flag.Parse()

	ctx := context.Background()

	resolvedLib := *libPath
	if *getLib != "" {
		staged, err := litertlm.LibFetch("", "", *getLib)
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
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	add, err := litertlm.RegisterTool(client, "calc_add",
		"Add two integers and return their sum.",
		func(ctx context.Context, in AddIn) (AddOut, error) {
			fmt.Printf("(invoke)  calc_add(a=%d, b=%d)\n", in.A, in.B)
			return AddOut{Sum: in.A + in.B}, nil
		})
	if err != nil {
		fmt.Fprintf(os.Stderr, "register tool: %v\n", err)
		os.Exit(1)
	}

	chat, err := client.NewChat(ctx,
		litertlm.WithSystemPrompt(systemPrompt),
		litertlm.WithTool(add),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new chat: %v\n", err)
		os.Exit(1)
	}
	defer chat.Close()

	fmt.Printf("user>     %s\n", *prompt)
	reply, err := chat.Send(ctx, *prompt)
	if err != nil {
		var hops *litertlm.ToolHopsError
		if errors.As(err, &hops) {
			fmt.Fprintf(os.Stderr, "hop cap exceeded after %d iterations; last reply: %s\n",
				hops.Hops, hops.LastReply.Raw())
		} else {
			fmt.Fprintf(os.Stderr, "send: %v\n", err)
		}
		os.Exit(1)
	}
	fmt.Printf("bot>      %s\n", reply.Text())
}
