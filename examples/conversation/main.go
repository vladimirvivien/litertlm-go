// conversation showcases the full tool-using flow on the high-level
// Chat API: declare a tool, send a user prompt, dispatch the
// structured tool_call the model returns, send the result back, read
// the final natural-language answer.
//
// See README.md in this directory for prerequisites and run instructions.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

const systemPrompt = "You are a calculator assistant. When the user asks for an arithmetic result, ALWAYS call the appropriate tool — never compute the answer in your head."

// calcAddTool returns the calc_add tool the model is allowed to call.
// It uses NewRawTool — the parameters map is the OpenAI / Anthropic
// function-calling schema, hand-built. For typed handlers and
// auto-dispatch, see RegisterTool.
func calcAddTool() *litertlm.RawTool {
	return litertlm.NewRawTool(
		"calc_add",
		"Add two integers and return their sum.",
		map[string]any{
			"type": "object",
			"properties": map[string]any{
				"a": map[string]any{"type": "integer", "description": "First addend"},
				"b": map[string]any{"type": "integer", "description": "Second addend"},
			},
			"required": []string{"a", "b"},
		},
	)
}

// executeToolCall dispatches on the function name and returns a value
// suitable for embedding directly as the tool message's `response`
// field — the chat template expects a JSON object, not a string.
func executeToolCall(call litertlm.ToolCall) (any, error) {
	switch call.Function.Name {
	case "calc_add":
		a, err := toInt(call.Function.Arguments["a"])
		if err != nil {
			return nil, fmt.Errorf("arg a: %w", err)
		}
		b, err := toInt(call.Function.Arguments["b"])
		if err != nil {
			return nil, fmt.Errorf("arg b: %w", err)
		}
		return map[string]int{"result": a + b}, nil
	default:
		return nil, fmt.Errorf("unknown tool: %s", call.Function.Name)
	}
}

// toInt extracts an int from arbitrary JSON-decoded values. Numeric
// arguments come across as float64 from encoding/json by default.
func toInt(v any) (int, error) {
	switch x := v.(type) {
	case float64:
		return int(x), nil
	case int:
		return x, nil
	case string:
		var n int
		if _, err := fmt.Sscanf(strings.TrimSpace(x), "%d", &n); err != nil {
			return 0, fmt.Errorf("parse %q: %w", x, err)
		}
		return n, nil
	default:
		return 0, fmt.Errorf("unsupported numeric type %T", v)
	}
}

func main() {
	model := flag.String("model", "", "path to .litertlm model file")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries (falls back to LITERTLM_LIB env)")
	prompt := flag.String("prompt", "What is 17 plus 25?", "user question")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "--model is required")
		os.Exit(2)
	}

	ctx := context.Background()
	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	client, err := litertlm.New(ctx,
		litertlm.WithLib(*libPath),
		litertlm.WithModel(*model),
		litertlm.WithBackend(*backend),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	chat, err := client.NewChat(ctx,
		litertlm.WithSystemPrompt(systemPrompt),
		litertlm.WithTool(calcAddTool()),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new chat: %v\n", err)
		os.Exit(1)
	}
	defer chat.Close()

	fmt.Printf("user>     %s\n", *prompt)
	reply, err := chat.Send(ctx, *prompt)
	if err != nil {
		fmt.Fprintf(os.Stderr, "send: %v\n", err)
		os.Exit(1)
	}

	if !reply.HasToolCalls() {
		// Model answered directly — no tool dispatch.
		fmt.Printf("bot>      %s\n", reply.Text())
		return
	}

	call := reply.ToolCalls()[0]
	fmt.Printf("(call)    %s(%v)\n", call.Function.Name, call.Function.Arguments)

	result, err := executeToolCall(call)
	if err != nil {
		// Surface the error back to the model so it can apologise/retry.
		result = map[string]string{"error": err.Error()}
	}
	fmt.Printf("(result)  %v\n", result)

	final, err := chat.SendToolResult(ctx, call.Function.Name, result)
	if err != nil {
		fmt.Fprintf(os.Stderr, "send tool result: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("bot>      %s\n", final.Text())
}
