// chat-history resumes a prior conversation by seeding Chat with a
// transcript of {role, content} messages via WithInitialMessages.
//
// See README.md in this directory for prerequisites and usage.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

var defaultHistory = []litertlm.Message{
	{Role: "user", Parts: []litertlm.Part{litertlm.Text("Hi, my name is Vlad and I'm building a Go binding for LiteRT-LM.")}},
	{Role: "assistant", Parts: []litertlm.Part{litertlm.Text("Hello Vlad. Are you using cgo or a pure-Go FFI approach?")}},
	{Role: "user", Parts: []litertlm.Part{litertlm.Text("Pure-Go via purego and jupiterrider/ffi. No cgo.")}},
	{Role: "assistant", Parts: []litertlm.Part{litertlm.Text("Nice — that avoids the cross-compilation headaches cgo would introduce.")}},
}

const defaultMessage = "What did I say my name was, and which FFI libraries am I using?"

func main() {
	model := flag.String("model", "", "path to .litertlm model file (required)")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.16.0)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	system := flag.String("system", "You are a concise assistant. Answer in one sentence.", "system prompt")
	historyPath := flag.String("history", "", "path to a JSON file containing the prior transcript (array of {role,content}); when empty, a built-in 4-turn transcript is used")
	contextPath := flag.String("context", "", "path to a JSON file with the extra-context object attached to the conversation preface; must decode to a JSON object")
	message := flag.String("message", defaultMessage, "new user message to send after history is seeded")
	filterChannels := flag.Bool("filter-channels", false, "exclude reasoning-channel (<|channel>...) tokens from the KV cache so they do not persist across turns")
	maxToolHops := flag.Int("max-tool-hops", 0, "cap on auto-dispatch hops; takes effect only when ManagedTools are registered (0 keeps the library default)")
	maxTokens := flag.Int("max-tokens", 4096, "engine total token budget (prompt + output)")
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

	history := defaultHistory
	if *historyPath != "" {
		h, err := loadHistory(*historyPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "load history: %v\n", err)
			os.Exit(1)
		}
		history = h
	}

	var extraContext string
	if *contextPath != "" {
		c, err := loadExtraContext(*contextPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "load extra context: %v\n", err)
			os.Exit(1)
		}
		extraContext = c
	}

	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	client, err := litertlm.New(ctx,
		litertlm.WithLib(resolvedLib),
		litertlm.WithModel(resolvedModel),
		litertlm.WithBackend(*backend),
		litertlm.WithMaxTokens(*maxTokens),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	chatOpts := []litertlm.ChatOption{
		litertlm.WithSystemPrompt(*system),
		litertlm.WithInitialMessages(history),
		litertlm.WithFilterChannelContentFromKVCache(*filterChannels),
	}
	if extraContext != "" {
		chatOpts = append(chatOpts, litertlm.WithExtraContext(extraContext))
	}
	if *maxToolHops > 0 {
		chatOpts = append(chatOpts, litertlm.WithMaxToolHops(*maxToolHops))
	}

	chat, err := client.NewChat(ctx, chatOpts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new chat: %v\n", err)
		os.Exit(1)
	}
	defer chat.Close()

	fmt.Println("=== Seeded history ===")
	for _, m := range history {
		var body string
		for _, p := range m.Parts {
			if p.IsText() {
				body = p.TextContent()
				break
			}
		}
		fmt.Printf("%-10s %s\n", m.Role+">", body)
	}
	fmt.Println()

	fmt.Printf("user>      %s\n", *message)
	reply, err := chat.Send(ctx, *message)
	if err != nil {
		fmt.Fprintf(os.Stderr, "send: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("assistant> %s\n", reply.Text())
}

func loadHistory(path string) ([]litertlm.Message, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	msgs := make([]litertlm.Message, len(raw))
	for i, r := range raw {
		msgs[i] = litertlm.Message{Role: r.Role, Parts: []litertlm.Part{litertlm.Text(r.Content)}}
	}
	return msgs, nil
}

// loadExtraContext reads path and verifies that the contents decode to a
// JSON object. The C side parses extra context with
// nlohmann::ordered_json::parse + is_object(); arrays, scalars, and free
// text are silently dropped.
func loadExtraContext(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	var obj map[string]any
	if err := json.Unmarshal(b, &obj); err != nil {
		return "", fmt.Errorf("%s: extra context must be a JSON object: %w", path, err)
	}
	return string(b), nil
}
