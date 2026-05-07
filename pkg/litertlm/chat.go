package litertlm

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"sync"
)

// ---- public types used by Chat ------------------------------------------

// ToolCall is a function invocation requested by the model. Returned
// by Reply.ToolCalls() when the assistant's turn is a tool-call
// rather than free text.
type ToolCall struct {
	Type     string           `json:"type"` // always "function"
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction holds the parsed name and arguments of a tool call.
// String-typed Argument values are normalised to plain Go strings
// (Gemma 4 quote markers are stripped on parse).
type ToolCallFunction struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// Message is one message in a conversation history. Used by
// WithInitialMessages to seed a Chat with prior turns.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ---- ChatOption -----------------------------------------------------------

// ChatOption configures a Chat at NewChat time.
type ChatOption func(*chatConfig)

type chatConfig struct {
	systemPrompt        string
	systemPromptSet     bool
	tools               []ToolDefinition
	initialMessages     []Message
	constrainedDecoding bool
}

// WithSystemPrompt sets the system message for the conversation. Pass
// just the content — the C side wraps it in a {role,content} envelope
// itself. (See c/engine.cc:litert_lm_conversation_create; passing the
// envelope here makes the chat template silently drop the prompt.)
func WithSystemPrompt(s string) ChatOption {
	return func(c *chatConfig) {
		c.systemPrompt = s
		c.systemPromptSet = true
	}
}

// WithTool attaches one or more ToolDefinitions to the chat. RawTool
// (built with NewRawTool) and ManagedTool[I, O] (built with
// RegisterTool) both satisfy ToolDefinition and may be mixed in the
// same call. The C side renders them into the model's native
// tool-declaration markers; ManagedTool entries also populate the
// chat's dispatch registry.
func WithTool(defs ...ToolDefinition) ChatOption {
	return func(c *chatConfig) {
		c.tools = append(c.tools, defs...)
	}
}

// WithInitialMessages seeds the conversation with a prior history.
// Messages are appended after the system prompt (if any) and before
// the first Send.
func WithInitialMessages(msgs []Message) ChatOption {
	return func(c *chatConfig) {
		c.initialMessages = append([]Message(nil), msgs...)
	}
}

// WithConstrainedDecoding toggles the engine's constrained-decoding
// mode. Only the boolean toggle is exposed by the C API today; schema
// delivery is upstream-pending.
func WithConstrainedDecoding(on bool) ChatOption {
	return func(c *chatConfig) { c.constrainedDecoding = on }
}

// ---- Chat -----------------------------------------------------------------

// Chat is a multi-turn conversation handle. It wraps a Conversation
// plus its ConversationConfig and surfaces high-level Send /
// SendStream / SendToolResult methods that absorb the JSON-marshaling
// boilerplate the low-level API requires.
type Chat struct {
	cfg  ConversationConfig
	conv Conversation

	mu     sync.Mutex
	closed bool
	tools  map[string]ToolDefinition // populated from WithTool defs
}

// NewChat creates a Chat rooted in the Client's engine. Caller must
// Close the returned Chat when done.
//
// Cancelling ctx during NewChat returns ctx.Err() to the caller
// promptly. The underlying C work has no cancel hook and runs to
// completion in the background; if it eventually succeeds after
// cancellation, the resulting handles are released automatically so
// nothing leaks.
func (c *Client) NewChat(ctx context.Context, opts ...ChatOption) (*Chat, error) {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return nil, fmt.Errorf("litertlm: Client is closed")
	}
	c.mu.Unlock()

	cfg := chatConfig{}
	for _, opt := range opts {
		opt(&cfg)
	}

	registry, err := buildToolRegistry(cfg.tools)
	if err != nil {
		return nil, fmt.Errorf("litertlm: NewChat: %w", err)
	}

	systemMessageJSON, err := encodeSystemPrompt(cfg)
	if err != nil {
		return nil, fmt.Errorf("litertlm: NewChat: %w", err)
	}
	toolsJSON, err := encodeTools(cfg.tools)
	if err != nil {
		return nil, fmt.Errorf("litertlm: NewChat: %w", err)
	}
	messagesJSON, err := encodeMessages(cfg.initialMessages)
	if err != nil {
		return nil, fmt.Errorf("litertlm: NewChat: %w", err)
	}

	return runCancellable(ctx,
		func() (*Chat, error) {
			return c.buildChat(systemMessageJSON, toolsJSON, messagesJSON, cfg.constrainedDecoding, registry)
		},
		func(ch *Chat) { _ = ch.Close() },
	)
}

// buildChat performs the synchronous C-side work of constructing a
// Chat. Split out so NewChat can run it under runCancellable.
func (c *Client) buildChat(systemMessageJSON, toolsJSON, messagesJSON string, constrainedDecoding bool, registry map[string]ToolDefinition) (*Chat, error) {
	convCfg, err := NewConversationConfig(c.engine, 0,
		systemMessageJSON, toolsJSON, messagesJSON, constrainedDecoding)
	if err != nil {
		return nil, fmt.Errorf("litertlm: NewChat: %w", err)
	}

	conv, err := c.engine.NewConversation(convCfg)
	if err != nil {
		convCfg.Delete()
		return nil, fmt.Errorf("litertlm: NewChat: %w", err)
	}

	return &Chat{cfg: convCfg, conv: conv, tools: registry}, nil
}

// buildToolRegistry walks defs and returns a name → def map for chat
// dispatch lookups. Errors when names collide or are empty.
func buildToolRegistry(defs []ToolDefinition) (map[string]ToolDefinition, error) {
	registry := map[string]ToolDefinition{}
	for _, d := range defs {
		name := d.Name()
		if name == "" {
			return nil, fmt.Errorf("tool with empty name")
		}
		if _, dup := registry[name]; dup {
			return nil, fmt.Errorf("duplicate tool name %q", name)
		}
		registry[name] = d
	}
	return registry, nil
}

// Close releases the underlying Conversation and ConversationConfig.
// Safe to call multiple times.
func (ch *Chat) Close() error {
	ch.mu.Lock()
	defer ch.mu.Unlock()
	if ch.closed {
		return nil
	}
	ch.closed = true
	ch.conv.Delete()
	ch.cfg.Delete()
	ch.conv = 0
	ch.cfg = 0
	return nil
}

// Send issues a user-role message and returns the assistant's Reply.
// Cancelling ctx aborts the in-flight call via Conversation.Cancel.
func (ch *Chat) Send(ctx context.Context, message string) (*Reply, error) {
	if err := ch.checkOpen(); err != nil {
		return nil, err
	}
	msgJSON, err := json.Marshal(map[string]string{"role": "user", "content": message})
	if err != nil {
		return nil, fmt.Errorf("litertlm: Send: marshal: %w", err)
	}
	return ch.send(ctx, string(msgJSON))
}

// SendStream issues a user-role message and returns an iterator over
// the streamed text chunks. Tool-using replies are surfaced as raw
// chunks in the stream — for structured tool_calls, prefer Send.
func (ch *Chat) SendStream(ctx context.Context, message string) iter.Seq2[Chunk, error] {
	return func(yield func(Chunk, error) bool) {
		if err := ch.checkOpen(); err != nil {
			yield(Chunk{}, err)
			return
		}
		msgJSON, err := json.Marshal(map[string]string{"role": "user", "content": message})
		if err != nil {
			yield(Chunk{}, fmt.Errorf("litertlm: SendStream: marshal: %w", err))
			return
		}

		stop := wireCancel(ctx, ch.conv.Cancel)
		defer stop()

		for sc := range ch.conv.SendMessageStreamCh(string(msgJSON), "") {
			ch := Chunk{Text: sc.Text, Final: sc.Final}
			if !yield(ch, sc.Err) {
				return
			}
			if sc.Err != nil {
				return
			}
		}
	}
}

// SendToolResult sends a tool-role message back to the model with the
// result of executing the tool named `name`. result is JSON-marshaled
// directly — pass a struct or map so the C-side template can render
// the response object faithfully.
func (ch *Chat) SendToolResult(ctx context.Context, name string, result any) (*Reply, error) {
	if err := ch.checkOpen(); err != nil {
		return nil, err
	}
	msgJSON, err := json.Marshal(map[string]any{
		"role": "tool",
		"content": []map[string]any{
			{"name": name, "response": result},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("litertlm: SendToolResult: marshal: %w", err)
	}
	return ch.send(ctx, string(msgJSON))
}

func (ch *Chat) send(ctx context.Context, msgJSON string) (*Reply, error) {
	stop := wireCancel(ctx, ch.conv.Cancel)
	defer stop()

	raw, err := ch.conv.SendMessage(msgJSON, "")
	if err != nil {
		if cerr := ctx.Err(); cerr != nil {
			return nil, cerr
		}
		return nil, fmt.Errorf("litertlm: send: %w", err)
	}
	return parseReply(raw)
}

func (ch *Chat) checkOpen() error {
	ch.mu.Lock()
	defer ch.mu.Unlock()
	if ch.closed {
		return fmt.Errorf("litertlm: Chat is closed")
	}
	return nil
}

// ---- helpers --------------------------------------------------------------

// encodeSystemPrompt JSON-encodes the system prompt as bare content.
// The C side wraps it in {role,content} itself — passing the envelope
// makes the chat template silently drop the system message.
func encodeSystemPrompt(cfg chatConfig) (string, error) {
	if !cfg.systemPromptSet {
		return "", nil
	}
	b, err := json.Marshal(cfg.systemPrompt)
	if err != nil {
		return "", fmt.Errorf("marshal system prompt: %w", err)
	}
	return string(b), nil
}

// encodeTools renders defs as the JSON tool-declaration list the C
// side expects. Each definition becomes:
//
//	{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
func encodeTools(defs []ToolDefinition) (string, error) {
	if len(defs) == 0 {
		return "", nil
	}
	list := make([]map[string]any, len(defs))
	for i, d := range defs {
		list[i] = map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        d.Name(),
				"description": d.Description(),
				"parameters":  d.Parameters(),
			},
		}
	}
	b, err := json.Marshal(list)
	if err != nil {
		return "", fmt.Errorf("marshal tools: %w", err)
	}
	return string(b), nil
}

func encodeMessages(msgs []Message) (string, error) {
	if len(msgs) == 0 {
		return "", nil
	}
	b, err := json.Marshal(msgs)
	if err != nil {
		return "", fmt.Errorf("marshal initial messages: %w", err)
	}
	return string(b), nil
}
