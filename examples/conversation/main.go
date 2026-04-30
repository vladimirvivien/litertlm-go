// conversation showcases the full Conversation API: system prompt +
// declared tools + structured tool_calls in the assistant reply +
// tool-role response message. The example runs entirely offline — the
// "tool" is just Go-side integer addition — so nothing else is needed
// beyond a chat-tuned .litertlm model.
//
// Flow:
//
//  1. Build a ConversationConfig with a system prompt and a tools_json
//     describing one function (calc_add).
//  2. Send the user question. The chat-tuned model emits a Gemma 4
//     <|tool_call> block; the C side parses it and surfaces it to us
//     as a structured `tool_calls` array on the assistant message.
//  3. Execute the call locally and send the result back as a tool-role
//     message — the chat template renders it as <|tool_response>...
//  4. The model continues the same model turn and produces the final
//     natural-language answer.
//
// See README.md in this directory for prerequisites and run instructions.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

const systemPrompt = "You are a calculator assistant. When the user asks for an arithmetic result, ALWAYS call the appropriate tool — never compute the answer in your head."

// ---- assistant-reply types ------------------------------------------------

// assistantMessage is the JSON shape returned by Conversation.SendMessage.
// Either Content (free-form text + multimodal parts) or ToolCalls (the
// model wants a tool invoked) may be populated; tool-using turns set the
// latter and leave Content empty.
type assistantMessage struct {
	Role      string        `json:"role"`
	Content   []contentPart `json:"content,omitempty"`
	ToolCalls []toolCall    `json:"tool_calls,omitempty"`
}

type contentPart struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type toolCall struct {
	Type     string       `json:"type"`
	Function toolCallFunc `json:"function"`
}

type toolCallFunc struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// ---- the only tool this example exposes -----------------------------------

// toolDefinitions is what gets passed as tools_json to the Conversation
// API. The shape mirrors the OpenAI / Anthropic function-calling schema,
// which the LiteRT-LM chat template then renders into Gemma 4's native
// <|tool>declaration:...<tool|> markers.
func toolDefinitions() []map[string]any {
	return []map[string]any{
		{
			"type": "function",
			"function": map[string]any{
				"name":        "calc_add",
				"description": "Add two integers and return their sum.",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"a": map[string]any{"type": "integer", "description": "First addend"},
						"b": map[string]any{"type": "integer", "description": "Second addend"},
					},
					"required": []string{"a", "b"},
				},
			},
		},
	}
}

// executeToolCall dispatches on the function name and returns a value
// suitable for embedding directly as the tool message's `response` field.
// The chat template expects that field to be a JSON object, so we return
// a map (not a JSON-encoded string).
func executeToolCall(call toolCall) (any, error) {
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

// toInt extracts an int from arbitrary JSON-decoded values. encoding/json
// turns numbers into float64 by default; the model occasionally also wraps
// numeric arguments in Gemma 4 native quote markers, so we tolerate
// strings too.
func toInt(v any) (int, error) {
	switch x := v.(type) {
	case float64:
		return int(x), nil
	case int:
		return x, nil
	case string:
		// Strip any leftover Gemma 4 quote markers before parsing.
		s := strings.ReplaceAll(x, `<|"|>`, "")
		var n int
		if _, err := fmt.Sscanf(s, "%d", &n); err != nil {
			return 0, fmt.Errorf("parse %q: %w", x, err)
		}
		return n, nil
	default:
		return 0, fmt.Errorf("unsupported numeric type %T", v)
	}
}

// ---- main -----------------------------------------------------------------

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

	if err := litertlm.Load(*libPath, *backend); err != nil {
		fmt.Fprintf(os.Stderr, "load: %v\n", err)
		os.Exit(1)
	}
	defer litertlm.Close()

	// Drop to LogInfo to see the C-side prompt rendering and decode logs.
	litertlm.SetMinLogLevel(litertlm.LogError)

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

	// systemMessageJSON expects just the content (the C side wraps it in
	// a {"role":"system","content":...} envelope itself); see the chat
	// example for the same caveat.
	sysJSON, _ := json.Marshal(systemPrompt)
	toolsJSON, _ := json.Marshal(toolDefinitions())

	cfg, err := litertlm.NewConversationConfig(engine, 0, string(sysJSON), string(toolsJSON), "", false)
	if err != nil {
		fmt.Fprintf(os.Stderr, "conv cfg: %v\n", err)
		os.Exit(1)
	}
	defer cfg.Delete()

	conv, err := engine.NewConversation(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "conv: %v\n", err)
		os.Exit(1)
	}
	defer conv.Delete()

	// ---- pass 1: send user prompt, expect a tool_call back ----
	userJSON, _ := json.Marshal(map[string]string{"role": "user", "content": *prompt})
	fmt.Printf("user>     %s\n", *prompt)

	rawReply, err := conv.SendMessage(string(userJSON), "")
	if err != nil {
		fmt.Fprintf(os.Stderr, "send: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("(raw)     %s\n", rawReply)

	var reply assistantMessage
	if err := json.Unmarshal([]byte(rawReply), &reply); err != nil {
		fmt.Fprintf(os.Stderr, "parse reply: %v\n", err)
		os.Exit(1)
	}

	if len(reply.ToolCalls) == 0 {
		// Model answered directly — print whatever text came back. This
		// branch is hit if the user prompt didn't warrant a tool call
		// (or the model decided to compute the answer in its head despite
		// the system instruction).
		fmt.Printf("bot>      %s\n", joinText(reply.Content))
		return
	}

	// ---- execute the tool ----
	call := reply.ToolCalls[0]
	fmt.Printf("(call)    %s(%v)\n", call.Function.Name, call.Function.Arguments)

	result, err := executeToolCall(call)
	if err != nil {
		// Surface the error back to the model so it can apologise/retry
		// rather than crashing the example.
		result = map[string]string{"error": err.Error()}
	}
	fmt.Printf("(result)  %v\n", result)

	// ---- pass 2: send the tool response, expect the final answer ----
	toolMsg, _ := json.Marshal(map[string]any{
		"role": "tool",
		"content": []map[string]any{
			{"name": call.Function.Name, "response": result},
		},
	})

	rawReply2, err := conv.SendMessage(string(toolMsg), "")
	if err != nil {
		fmt.Fprintf(os.Stderr, "send tool response: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("(raw)     %s\n", rawReply2)

	var reply2 assistantMessage
	if err := json.Unmarshal([]byte(rawReply2), &reply2); err != nil {
		fmt.Fprintf(os.Stderr, "parse final reply: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("bot>      %s\n", joinText(reply2.Content))
}

func joinText(parts []contentPart) string {
	var b strings.Builder
	for _, p := range parts {
		if p.Type == "text" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}
