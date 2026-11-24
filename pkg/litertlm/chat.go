package litertlm

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"slices"
	"sync"
)

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

// Message is one message in a conversation history used by WithInitialMessages.
type Message struct {
	Role  string
	Parts []Part
}

// ChatOption configures a Chat at NewChat time.
type ChatOption func(*chatConfig)

type chatConfig struct {
	systemPrompt                    string
	systemPromptSet                 bool
	tools                           []ToolDefinition
	initialMessages                 []Message
	constrainedDecoding             bool
	constraintProvider              *ConstraintProviderType
	promptTemplate                  string
	thinking                        *ThinkingConfig
	maxToolHops                     int
	extraContextJSON                string
	filterChannelContentFromKVCache *bool
}

// WithSystemPrompt sets the system message for the conversation. Pass
// just the content — the C side wraps it in a {role,content} envelope itself.
func WithSystemPrompt(s string) ChatOption {
	return func(c *chatConfig) {
		c.systemPrompt = s
		c.systemPromptSet = true
	}
}

// WithTool attaches one or more ToolDefinitions to the chat.
func WithTool(defs ...ToolDefinition) ChatOption {
	return func(c *chatConfig) {
		c.tools = append(c.tools, defs...)
	}
}

// WithInitialMessages seeds the conversation with a prior history appended
// after any system prompt and before the first Send.
func WithInitialMessages(msgs []Message) ChatOption {
	return func(c *chatConfig) {
		c.initialMessages = slices.Clone(msgs)
	}
}

// WithConstrainedDecoding toggles the engine's constrained-decoding mode.
func WithConstrainedDecoding(on bool) ChatOption {
	return func(c *chatConfig) { c.constrainedDecoding = on }
}

// WithConstraintProvider selects the constrained-decoding grammar provider.
func WithConstraintProvider(provider ConstraintProviderType) ChatOption {
	return func(c *chatConfig) { c.constraintProvider = new(provider) }
}

// WithPromptTemplate overrides the model's default chat prompt template.
func WithPromptTemplate(tmpl string) ChatOption {
	return func(c *chatConfig) { c.promptTemplate = tmpl }
}

// WithThinking sets conversation-wide thinking/reasoning configuration.
func WithThinking(cfg ThinkingConfig) ChatOption {
	return func(c *chatConfig) { c.thinking = new(cfg) }
}

// WithExtraContext attaches an extra-context JSON string used in the
// conversation preface. Empty string is a no-op.
func WithExtraContext(extraContextJSON string) ChatOption {
	return func(c *chatConfig) { c.extraContextJSON = extraContextJSON }
}

// WithFilterChannelContentFromKVCache toggles whether reasoning-channel
// tokens (<|channel> ... <channel|>) are excluded from the KV cache.
func WithFilterChannelContentFromKVCache(on bool) ChatOption {
	return func(c *chatConfig) { c.filterChannelContentFromKVCache = new(on) }
}

// WithMaxToolHops caps the number of dispatch iterations Chat.Send runs
// when ManagedTools are registered. n <= 0 keeps the default of defaultMaxToolHops.
func WithMaxToolHops(n int) ChatOption {
	return func(c *chatConfig) {
		if n > 0 {
			c.maxToolHops = n
		}
	}
}

// Chat is a multi-turn conversation handle. It wraps a Conversation plus
// its ConversationConfig and surfaces high-level Send / SendStream /
// SendToolResult methods.
type Chat struct {
	cfg  ConversationConfig
	conv Conversation

	mu             sync.Mutex
	closed         bool
	tools          map[string]ToolDefinition
	maxToolHops    int
	singleThreaded bool
}

type chatBuildSpec struct {
	cfg               chatConfig
	systemMessageJSON string
	toolsJSON         string
	messagesJSON      string
	registry          map[string]ToolDefinition
}

// NewChat creates a Chat rooted in the Client's engine. Caller must Close
// the returned Chat when done.
func (c *Client) NewChat(ctx context.Context, opts ...ChatOption) (*Chat, error) {
	if _, err := c.openEngine(); err != nil {
		return nil, err
	}

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

	spec := chatBuildSpec{
		cfg:               cfg,
		systemMessageJSON: systemMessageJSON,
		toolsJSON:         toolsJSON,
		messagesJSON:      messagesJSON,
		registry:          registry,
	}

	return runCancellable(ctx,
		func() (*Chat, error) { return c.buildChat(spec) },
		func(ch *Chat) { _ = ch.Close() },
	)
}

func (c *Client) buildChat(spec chatBuildSpec) (*Chat, error) {
	var sessionConfig SessionConfig
	if c.cfg.defaultSampler != nil {
		sc, err := NewSessionConfig()
		if err != nil {
			return nil, fmt.Errorf("litertlm: NewChat: session config: %w", err)
		}
		defer sc.Delete()
		if err = sc.SetSamplerParams(*c.cfg.defaultSampler); err != nil {
			return nil, fmt.Errorf("litertlm: NewChat: session config: %w", err)
		}
		sessionConfig = sc
	}

	engine, err := c.openEngine()
	if err != nil {
		return nil, fmt.Errorf("litertlm: NewChat: %w", err)
	}

	convCfg, err := NewConversationConfig(engine, sessionConfig,
		spec.systemMessageJSON, spec.toolsJSON, spec.messagesJSON, spec.cfg.constrainedDecoding)
	if err != nil {
		return nil, fmt.Errorf("litertlm: NewChat: %w", err)
	}

	if spec.cfg.promptTemplate != "" {
		if err = convCfg.SetPromptTemplate(spec.cfg.promptTemplate); err != nil {
			convCfg.Delete()
			return nil, fmt.Errorf("litertlm: NewChat: %w", err)
		}
	}
	if spec.cfg.constraintProvider != nil {
		if err = convCfg.SetConstraintProvider(*spec.cfg.constraintProvider); err != nil {
			convCfg.Delete()
			return nil, fmt.Errorf("litertlm: NewChat: %w", err)
		}
	}
	if spec.cfg.thinking != nil {
		if err = convCfg.SetThinkingConfig(*spec.cfg.thinking); err != nil {
			convCfg.Delete()
			return nil, fmt.Errorf("litertlm: NewChat: %w", err)
		}
	}
	if err = convCfg.SetExtraContext(spec.cfg.extraContextJSON); err != nil {
		convCfg.Delete()
		return nil, fmt.Errorf("litertlm: NewChat: %w", err)
	}
	if spec.cfg.filterChannelContentFromKVCache != nil {
		if err = convCfg.SetFilterChannelContentFromKVCache(*spec.cfg.filterChannelContentFromKVCache); err != nil {
			convCfg.Delete()
			return nil, fmt.Errorf("litertlm: NewChat: %w", err)
		}
	}

	conv, err := engine.NewConversation(convCfg)
	if err != nil {
		convCfg.Delete()
		return nil, fmt.Errorf("litertlm: NewChat: %w", err)
	}

	return &Chat{
		cfg:            convCfg,
		conv:           conv,
		tools:          spec.registry,
		maxToolHops:    spec.cfg.maxToolHops,
		singleThreaded: c.singleThreaded(),
	}, nil
}

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

// Conversation returns the underlying Conversation handle, or 0 if closed.
func (ch *Chat) Conversation() Conversation {
	ch.mu.Lock()
	defer ch.mu.Unlock()
	return ch.conv
}

func (ch *Chat) openConv() (Conversation, error) {
	ch.mu.Lock()
	defer ch.mu.Unlock()
	if ch.closed {
		return 0, fmt.Errorf("litertlm: Chat is closed")
	}
	return ch.conv, nil
}

// Clone returns a new Chat whose underlying Conversation mirrors ch's
// prefilled state. ch must outlive every clone derived from it.
func (ch *Chat) Clone() (*Chat, error) {
	ch.mu.Lock()
	if ch.closed {
		ch.mu.Unlock()
		return nil, fmt.Errorf("litertlm: Chat is closed")
	}
	conv := ch.conv
	tools := ch.tools
	hops := ch.maxToolHops
	singleThreaded := ch.singleThreaded
	ch.mu.Unlock()

	clonedConv, err := conv.Clone()
	if err != nil {
		return nil, fmt.Errorf("litertlm: Chat.Clone: %w", err)
	}
	return &Chat{
		conv:           clonedConv,
		tools:          tools,
		maxToolHops:    hops,
		singleThreaded: singleThreaded,
	}, nil
}

// TokenCount returns the number of tokens currently held in this Chat's
// underlying conversation KV cache (prefill + decode).
func (ch *Chat) TokenCount() (int, error) {
	conv, err := ch.openConv()
	if err != nil {
		return 0, err
	}
	n, err := conv.TokenCount()
	if err != nil {
		return 0, fmt.Errorf("litertlm: Chat.TokenCount: %w", err)
	}
	return n, nil
}

// Send issues a user-role message and returns the assistant's Reply,
// running any registered ManagedTool dispatch loop automatically.
func (ch *Chat) Send(ctx context.Context, message string, opts ...RuntimeOption) (*Reply, error) {
	conv, err := ch.openConv()
	if err != nil {
		return nil, err
	}
	msgJSON, err := json.Marshal(map[string]string{"role": "user", "content": message})
	if err != nil {
		return nil, fmt.Errorf("litertlm: Send: marshal: %w", err)
	}
	return ch.send(ctx, conv, string(msgJSON), resolveRuntimeConfig(opts))
}

type singleThreadedConv struct{ Conversation }

func (s singleThreadedConv) SendMessageStreamCh(messageJSON, extraContext string, opts OptionalArgs) <-chan StreamChunk {
	return s.sendMessageStreamCh(messageJSON, extraContext, opts, true)
}

func (ch *Chat) streamTransport(conv Conversation) chatTransport {
	if ch.singleThreaded {
		return singleThreadedConv{conv}
	}
	return conv
}

// SendStream issues a user-role message and returns an iterator over
// streamed text chunks, transparently running ManagedTool dispatch hops.
func (ch *Chat) SendStream(ctx context.Context, message string, opts ...RuntimeOption) iter.Seq2[Chunk, error] {
	return func(yield func(Chunk, error) bool) {
		conv, err := ch.openConv()
		if err != nil {
			yield(Chunk{}, err)
			return
		}
		msgJSON, err := json.Marshal(map[string]string{"role": "user", "content": message})
		if err != nil {
			yield(Chunk{}, fmt.Errorf("litertlm: SendStream: marshal: %w", err))
			return
		}
		ch.streamWithDispatch(ctx, ch.streamTransport(conv), string(msgJSON), resolveRuntimeConfig(opts), yield)
	}
}

// SendMulti issues a multimodal user-role message and returns the assistant's Reply.
func (ch *Chat) SendMulti(ctx context.Context, parts []Part, opts ...RuntimeOption) (*Reply, error) {
	conv, err := ch.openConv()
	if err != nil {
		return nil, err
	}
	if len(parts) == 0 {
		return nil, fmt.Errorf("litertlm: SendMulti: empty parts")
	}
	msgJSON, err := partsToConversationMessage(parts)
	if err != nil {
		return nil, fmt.Errorf("litertlm: SendMulti: %w", err)
	}
	return ch.send(ctx, conv, msgJSON, resolveRuntimeConfig(opts))
}

// SendMultiStream is the streaming sibling of SendMulti.
func (ch *Chat) SendMultiStream(ctx context.Context, parts []Part, opts ...RuntimeOption) iter.Seq2[Chunk, error] {
	return func(yield func(Chunk, error) bool) {
		conv, err := ch.openConv()
		if err != nil {
			yield(Chunk{}, err)
			return
		}
		if len(parts) == 0 {
			yield(Chunk{}, fmt.Errorf("litertlm: SendMultiStream: empty parts"))
			return
		}
		msgJSON, err := partsToConversationMessage(parts)
		if err != nil {
			yield(Chunk{}, fmt.Errorf("litertlm: SendMultiStream: %w", err))
			return
		}
		ch.streamWithDispatch(ctx, ch.streamTransport(conv), msgJSON, resolveRuntimeConfig(opts), yield)
	}
}

// SendToolResult sends a tool-role message back to the model with the
// result of executing the tool named `name`.
func (ch *Chat) SendToolResult(ctx context.Context, name string, result any, opts ...RuntimeOption) (*Reply, error) {
	conv, err := ch.openConv()
	if err != nil {
		return nil, err
	}
	msgJSON, err := encodeToolResults([]toolResult{{name: name, response: result}})
	if err != nil {
		return nil, fmt.Errorf("litertlm: SendToolResult: %w", err)
	}
	return ch.send(ctx, conv, msgJSON, resolveRuntimeConfig(opts))
}
