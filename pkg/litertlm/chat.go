package litertlm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"sort"
	"strings"
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
// WithInitialMessages to seed a Chat with prior turns. Parts may
// carry text, image, and audio content in the same []Part shape
// accepted by SendMulti; pure-text history uses
// []Part{Text("...")}.
type Message struct {
	Role  string
	Parts []Part
}

// ---- ChatOption -----------------------------------------------------------

// ChatOption configures a Chat at NewChat time.
type ChatOption func(*chatConfig)

type chatConfig struct {
	systemPrompt                    string
	systemPromptSet                 bool
	tools                           []ToolDefinition
	initialMessages                 []Message
	constrainedDecoding             bool
	maxToolHops                     int
	extraContextJSON                string
	filterChannelContentFromKVCache *bool
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
// the first Send. Each Message carries Role plus a []Part body —
// text-only history uses []Part{Text("...")}; multimodal history
// may include Image / Audio parts the same way SendMulti accepts
// them.
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

// WithExtraContext attaches an extra-context JSON string used in the
// conversation preface. Empty string is a no-op.
func WithExtraContext(extraContextJSON string) ChatOption {
	return func(c *chatConfig) { c.extraContextJSON = extraContextJSON }
}

// WithFilterChannelContentFromKVCache toggles whether the model's
// reasoning-channel tokens (those framed by <|channel> ... <channel|>)
// are excluded from the conversation's KV cache. When on, the
// reasoning content does not persist across turns.
func WithFilterChannelContentFromKVCache(on bool) ChatOption {
	return func(c *chatConfig) { c.filterChannelContentFromKVCache = &on }
}

// WithMaxToolHops caps the number of dispatch iterations Chat.Send
// runs when ManagedTools are registered. Each iteration is one
// model→tool→model round-trip. Exceeding the cap returns an error
// matching ErrToolHopsExceeded; use errors.As(err, &*ToolHopsError)
// to inspect the partial last reply. n <= 0 keeps the default of
// defaultMaxToolHops.
func WithMaxToolHops(n int) ChatOption {
	return func(c *chatConfig) {
		if n > 0 {
			c.maxToolHops = n
		}
	}
}

// ---- Chat -----------------------------------------------------------------

// defaultMaxToolHops is the dispatch-loop iteration cap when
// WithMaxToolHops isn't supplied.
const defaultMaxToolHops = 5

// chatTransport is the conversation-side seam used by Chat.send,
// Chat.streamWithDispatch, and the auto-dispatch loop. The real
// Conversation handle satisfies it; tests inject a stub.
type chatTransport interface {
	SendMessage(messageJSON, extraContext string, opts OptionalArgs) (string, error)
	SendMessageStreamCh(messageJSON, extraContext string, opts OptionalArgs) <-chan StreamChunk
	Cancel()
}

// Compile-time check: the real Conversation handle satisfies the
// transport interface.
var _ chatTransport = Conversation(0)

// ErrToolHopsExceeded matches errors returned when Chat.Send exceeds
// the dispatch hop cap. Use errors.As to a *ToolHopsError to inspect
// the partial last reply.
var ErrToolHopsExceeded = errors.New("litertlm: tool hops exceeded")

// ToolHopsError carries the last assistant reply and the cap that
// was reached. Returned wrapped in errors.Is(err, ErrToolHopsExceeded)
// matches.
type ToolHopsError struct {
	LastReply *Reply
	Hops      int
}

func (e *ToolHopsError) Error() string {
	return fmt.Sprintf("litertlm: tool hops exceeded (cap %d)", e.Hops)
}

func (e *ToolHopsError) Unwrap() error { return ErrToolHopsExceeded }

// Chat is a multi-turn conversation handle. It wraps a Conversation
// plus its ConversationConfig and surfaces high-level Send /
// SendStream / SendToolResult methods that absorb the JSON-marshaling
// boilerplate the low-level API requires.
type Chat struct {
	cfg  ConversationConfig
	conv Conversation

	mu          sync.Mutex
	closed      bool
	tools       map[string]ToolDefinition // populated from WithTool defs
	maxToolHops int                       // 0 → defaultMaxToolHops
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
			return c.buildChat(systemMessageJSON, toolsJSON, messagesJSON,
				cfg.constrainedDecoding, registry, cfg.maxToolHops,
				cfg.extraContextJSON, cfg.filterChannelContentFromKVCache)
		},
		func(ch *Chat) { _ = ch.Close() },
	)
}

// buildChat performs the synchronous C-side work of constructing a
// Chat. Split out so NewChat can run it under runCancellable.
func (c *Client) buildChat(systemMessageJSON, toolsJSON, messagesJSON string,
	constrainedDecoding bool, registry map[string]ToolDefinition, maxToolHops int,
	extraContextJSON string, filterChannelContentFromKVCache *bool,
) (*Chat, error) {
	// If the client has a defaultSampler, materialize a SessionConfig
	// carrying it so the Conversation uses the same sampler the Generate
	// path does. Without this, the Conversation falls back to the C
	// engine's built-in greedy default, which on small models traps the
	// decoder in self-reinforcing token loops. The C set_session_config
	// copies the underlying struct (c/engine.cc:276), so the local handle
	// is safe to release immediately after NewConversationConfig returns.
	var sessionConfig SessionConfig
	if c.cfg.defaultSampler != nil {
		sc, err := NewSessionConfig()
		if err != nil {
			return nil, fmt.Errorf("litertlm: NewChat: session config: %w", err)
		}
		defer sc.Delete()
		sc.SetSamplerParams(*c.cfg.defaultSampler)
		sessionConfig = sc
	}

	convCfg, err := NewConversationConfig(c.engine, sessionConfig,
		systemMessageJSON, toolsJSON, messagesJSON, constrainedDecoding)
	if err != nil {
		return nil, fmt.Errorf("litertlm: NewChat: %w", err)
	}

	if err = convCfg.SetExtraContext(extraContextJSON); err != nil {
		convCfg.Delete()
		return nil, fmt.Errorf("litertlm: NewChat: %w", err)
	}
	if filterChannelContentFromKVCache != nil {
		convCfg.SetFilterChannelContentFromKVCache(*filterChannelContentFromKVCache)
	}

	conv, err := c.engine.NewConversation(convCfg)
	if err != nil {
		convCfg.Delete()
		return nil, fmt.Errorf("litertlm: NewChat: %w", err)
	}

	return &Chat{cfg: convCfg, conv: conv, tools: registry, maxToolHops: maxToolHops}, nil
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

// Clone returns a new Chat whose underlying Conversation mirrors ch's
// prefilled state — activation frames and KV cache included. Useful
// for branching tool loops (run N candidate tools off one prefilled
// prompt) and for structured-output retries where each attempt should
// start from the same conversation history.
//
// The cloned Chat shares ch's tool registry and dispatch-hop limit.
// The ConversationConfig is owned by ch; the clone holds none and
// only releases its Conversation on Close. ch must outlive every
// clone derived from it; closing ch invalidates the underlying
// configuration that backs each clone's Conversation.
//
// Clone fails when the underlying executor does not implement
// Session.Clone. As of LiteRT-LM v0.12.0 this is the case for the
// LiteRT executor used by Gemma 4 on CPU and GPU.
func (ch *Chat) Clone() (*Chat, error) {
	ch.mu.Lock()
	if ch.closed {
		ch.mu.Unlock()
		return nil, fmt.Errorf("litertlm: Chat is closed")
	}
	conv := ch.conv
	tools := ch.tools
	hops := ch.maxToolHops
	ch.mu.Unlock()

	clonedConv, err := conv.Clone()
	if err != nil {
		return nil, fmt.Errorf("litertlm: Chat.Clone: %w", err)
	}
	return &Chat{
		conv:        clonedConv,
		tools:       tools,
		maxToolHops: hops,
	}, nil
}

// TokenCount returns the number of tokens currently held in this Chat's
// underlying conversation KV cache (prefill + decode, accumulated across
// every turn including tool-dispatch hops). Use it to project a chat's
// size against the engine's max-token budget.
//
// It does not require WithBenchmarkEnabled. For a per-turn prefill /
// decode breakdown, read Conversation.BenchmarkInfo() instead (which
// does require benchmark collection).
//
// Requires LiteRT-LM v0.13.1 or newer.
func (ch *Chat) TokenCount() (int, error) {
	ch.mu.Lock()
	closed := ch.closed
	conv := ch.conv
	ch.mu.Unlock()
	if closed {
		return 0, fmt.Errorf("litertlm: Chat is closed")
	}
	n, err := conv.TokenCount()
	if err != nil {
		return 0, fmt.Errorf("litertlm: Chat.TokenCount: %w", err)
	}
	return n, nil
}

// Send issues a user-role message and returns the assistant's Reply.
// When the chat has at least one ManagedTool registered and the
// reply contains tool calls that all map to dispatchable entries,
// Send runs the tool-call → invoke → tool-result → next-turn loop
// until the model produces a text-only reply (or the dispatch cap is
// reached). Replies containing any non-dispatchable tool call (a
// RawTool or an unknown name) are returned as-is for manual handling
// with Reply.ToolCalls() + Chat.SendToolResult. WithReturnToolRequests
// forces this manual-handling path even for fully dispatchable calls.
//
// Per-call opts apply to this turn (and every dispatch hop it
// triggers). Text-only turns ignore multimodal knobs such as
// WithVisualTokenBudget.
//
// Cancelling ctx aborts the in-flight call via Conversation.Cancel.
func (ch *Chat) Send(ctx context.Context, message string, opts ...RuntimeOption) (*Reply, error) {
	if err := ch.checkOpen(); err != nil {
		return nil, err
	}
	msgJSON, err := json.Marshal(map[string]string{"role": "user", "content": message})
	if err != nil {
		return nil, fmt.Errorf("litertlm: Send: marshal: %w", err)
	}
	return ch.send(ctx, ch.conv, string(msgJSON), resolveRuntimeConfig(opts))
}

// SendStream issues a user-role message and returns an iterator over
// the streamed text chunks. Each chunk's Text is the assistant's text
// for that step — the per-chunk JSON envelope the C side surfaces is
// parsed and only the inner content is yielded.
//
// When ManagedTools are registered and the model emits dispatchable
// tool_calls during the stream, SendStream runs the dispatch loop
// transparently: tool handlers are invoked, results are fed back to
// the model, and streaming resumes on the post-tool reply. The
// caller sees one contiguous text stream and one trailing Final
// chunk, with text from pre-dispatch turns interleaved with text
// from post-dispatch turns. Replies containing any
// non-dispatchable tool call (a RawTool or an unknown name) end
// the stream with an error.
//
// Per-call opts apply to this turn (and every dispatch hop it
// triggers). Text-only turns ignore multimodal knobs such as
// WithVisualTokenBudget.
func (ch *Chat) SendStream(ctx context.Context, message string, opts ...RuntimeOption) iter.Seq2[Chunk, error] {
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
		ch.streamWithDispatch(ctx, ch.conv, string(msgJSON), resolveRuntimeConfig(opts), yield)
	}
}

// extractStreamChunkText pulls the assistant text out of a single
// streaming chunk's JSON envelope. The C side emits chunks shaped like
// `{"role":"assistant","content":[{"type":"text","text":"..."}]}` per
// token. Returns the raw chunk unchanged if it isn't valid JSON in
// that shape — preserves anything the C side might surface that the
// wrapper does not yet model.
func extractStreamChunkText(raw string) string {
	if raw == "" {
		return ""
	}
	var msg struct {
		Content []replyContentPart `json:"content"`
	}
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		return raw
	}
	if len(msg.Content) == 0 {
		return ""
	}
	var b strings.Builder
	for _, p := range msg.Content {
		if p.Type == "text" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// SendMulti issues a multimodal user-role message and returns the
// assistant's Reply. Tool dispatch behaves identically to Send;
// WithReturnToolRequests bypasses dispatch the same way.
//
// Image / audio Parts require the Client's WithVisionBackend /
// WithAudioBackend at New time. A []Part containing only text Parts
// is equivalent to Send with the text concatenated.
//
// The Chat's underlying Conversation accumulates KV state across
// turns, including image and audio embeddings — follow-up text
// turns can reference earlier multimodal content.
//
// Per-call opts apply to this turn (and every dispatch hop it
// triggers). WithVisualTokenBudget(n) caps the number of vision
// tokens consumed on this turn.
//
// Cancelling ctx aborts the in-flight call via Conversation.Cancel.
func (ch *Chat) SendMulti(ctx context.Context, parts []Part, opts ...RuntimeOption) (*Reply, error) {
	if err := ch.checkOpen(); err != nil {
		return nil, err
	}
	if len(parts) == 0 {
		return nil, fmt.Errorf("litertlm: SendMulti: empty parts")
	}
	msgJSON, err := partsToConversationMessage(parts)
	if err != nil {
		return nil, fmt.Errorf("litertlm: SendMulti: %w", err)
	}
	return ch.send(ctx, ch.conv, msgJSON, resolveRuntimeConfig(opts))
}

// SendMultiStream is the streaming sibling of SendMulti. Chunk.Text
// carries bare text; the multimodal stream's JSON envelope is
// parsed internally. ManagedTool calls emitted by the model during
// the stream are auto-dispatched the same way SendStream handles
// them (see SendStream godoc for the dispatch semantics).
//
// Per-call opts apply to this turn (and every dispatch hop it
// triggers). WithVisualTokenBudget(n) caps the number of vision
// tokens consumed on this turn.
func (ch *Chat) SendMultiStream(ctx context.Context, parts []Part, opts ...RuntimeOption) iter.Seq2[Chunk, error] {
	return func(yield func(Chunk, error) bool) {
		if err := ch.checkOpen(); err != nil {
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
		ch.streamWithDispatch(ctx, ch.conv, msgJSON, resolveRuntimeConfig(opts), yield)
	}
}

// streamWithDispatch drives the streaming auto-dispatch loop. Each
// chunk from the transport is parsed for content text and tool_calls;
// text is yielded immediately, tool_calls are accumulated, and on
// stream completion the dispatcher invokes the handlers and resumes
// streaming the post-tool turn. Intermediate Final markers from
// inner turns are suppressed; one synthetic Final yields at the
// very end after all dispatch is finished.
//
// transport is parameterized so tests inject a stub satisfying
// chatTransport. cfg carries per-call knobs; OptionalArgs are
// materialized once here and reused across every dispatch hop.
func (ch *Chat) streamWithDispatch(ctx context.Context, transport chatTransport, msgJSON string, cfg runtimeConfig, yield func(Chunk, error) bool) {
	optArgs, err := buildOptionalArgs(cfg)
	if err != nil {
		yield(Chunk{}, err)
		return
	}
	defer optArgs.Delete()

	cap := ch.maxToolHops
	if cap <= 0 {
		cap = defaultMaxToolHops
	}

	for hop := 0; ; hop++ {
		var calls []ToolCall
		cancelled, streamErr := ch.streamOne(ctx, transport, msgJSON, optArgs, &calls, yield)
		if cancelled || streamErr != nil {
			if streamErr != nil {
				yield(Chunk{}, streamErr)
			}
			return
		}

		if len(calls) == 0 {
			// No tool calls — final turn complete. Yield the
			// synthetic Final to close the iterator.
			yield(Chunk{Text: "", Final: true}, nil)
			return
		}

		reply := &Reply{toolCalls: calls}
		if hop >= cap {
			yield(Chunk{}, &ToolHopsError{LastReply: reply, Hops: cap})
			return
		}
		results, err := ch.invokeOrInform(ctx, calls, cfg.maxConcurrentTools)
		if err != nil {
			yield(Chunk{}, err)
			return
		}
		msgJSON, err = encodeToolResults(results)
		if err != nil {
			yield(Chunk{}, err)
			return
		}
	}
}

// streamOne consumes one turn from the transport's stream channel.
// Text chunks are yielded with Final=false (the synthetic Final is
// emitted by streamWithDispatch at the end of the dispatch loop).
// Tool calls observed in any chunk are appended to *calls. Returns
// (cancelled, err): cancelled=true when the caller's yield returned
// false; err is set on a transport-side error.
func (ch *Chat) streamOne(ctx context.Context, transport chatTransport, msgJSON string, opts OptionalArgs, calls *[]ToolCall, yield func(Chunk, error) bool) (bool, error) {
	stop := wireCancel(ctx, transport.Cancel)
	defer stop()

	for sc := range transport.SendMessageStreamCh(msgJSON, "", opts) {
		if sc.Err != nil {
			return false, sc.Err
		}
		text, chunkCalls := extractStreamEnvelope(sc.Text)
		if len(chunkCalls) > 0 {
			*calls = append(*calls, chunkCalls...)
		}
		if text == "" {
			continue
		}
		if !yield(Chunk{Text: text, Final: false}, nil) {
			return true, nil
		}
	}
	return false, nil
}

// extractStreamEnvelope parses one streamed chunk envelope. Returns
// the concatenated text from content[] (empty when the chunk has no
// text parts) and any tool_calls present in the chunk. A chunk that
// is not a parseable JSON envelope returns (raw, nil) so unrecognised
// surfaces still reach the caller as text.
func extractStreamEnvelope(raw string) (string, []ToolCall) {
	if raw == "" {
		return "", nil
	}
	var msg struct {
		Content   []replyContentPart `json:"content,omitempty"`
		ToolCalls []ToolCall         `json:"tool_calls,omitempty"`
	}
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		return raw, nil
	}
	for i := range msg.ToolCalls {
		args := msg.ToolCalls[i].Function.Arguments
		for k, v := range args {
			if s, ok := v.(string); ok {
				args[k] = strings.ReplaceAll(s, `<|"|>`, "")
			}
		}
	}
	var text strings.Builder
	for _, p := range msg.Content {
		if p.Type == "text" {
			text.WriteString(p.Text)
		}
	}
	return text.String(), msg.ToolCalls
}

// SendToolResult sends a tool-role message back to the model with the
// result of executing the tool named `name`. result is JSON-marshaled
// directly — pass a struct or map so the C-side template can render
// the response object faithfully.
//
// Like Send, SendToolResult enters the auto-dispatch loop when the
// model's reply contains tool calls all mapping to ManagedTools.
// WithReturnToolRequests bypasses dispatch and returns the reply
// for manual handling. Per-call opts apply to this turn (and every
// dispatch hop it triggers). Tool-result turns carry no images, so
// WithVisualTokenBudget is a no-op here.
func (ch *Chat) SendToolResult(ctx context.Context, name string, result any, opts ...RuntimeOption) (*Reply, error) {
	if err := ch.checkOpen(); err != nil {
		return nil, err
	}
	msgJSON, err := encodeToolResults([]toolResult{{name: name, response: result}})
	if err != nil {
		return nil, fmt.Errorf("litertlm: SendToolResult: %w", err)
	}
	return ch.send(ctx, ch.conv, msgJSON, resolveRuntimeConfig(opts))
}

// toolResult holds one tool's response within a tool-role message.
type toolResult struct {
	name     string
	response any
}

// encodeToolResults marshals one or more tool results into a single
// tool-role message JSON envelope.
func encodeToolResults(results []toolResult) (string, error) {
	content := make([]map[string]any, len(results))
	for i, r := range results {
		content[i] = map[string]any{"name": r.name, "response": r.response}
	}
	b, err := json.Marshal(map[string]any{
		"role":    "tool",
		"content": content,
	})
	if err != nil {
		return "", fmt.Errorf("marshal tool results: %w", err)
	}
	return string(b), nil
}

// send drives the dispatch loop. transport is parameterized so tests
// can inject a stub satisfying chatTransport. cfg carries per-call
// knobs (visual token budget, return-tool-requests flag, dispatch
// concurrency); OptionalArgs are materialized once here and reused
// across every dispatch hop.
func (ch *Chat) send(ctx context.Context, transport chatTransport, msgJSON string, cfg runtimeConfig) (*Reply, error) {
	optArgs, err := buildOptionalArgs(cfg)
	if err != nil {
		return nil, err
	}
	defer optArgs.Delete()

	cap := ch.maxToolHops
	if cap <= 0 {
		cap = defaultMaxToolHops
	}

	for hop := 0; ; hop++ {
		reply, err := sendOne(ctx, transport, msgJSON, optArgs)
		if err != nil {
			return nil, err
		}
		if !reply.HasToolCalls() || !ch.allCallsDispatchable(reply) || cfg.returnToolRequests {
			return reply, nil
		}
		if hop >= cap {
			return nil, &ToolHopsError{LastReply: reply, Hops: cap}
		}
		results, err := ch.invokeAll(ctx, reply.ToolCalls(), cfg.maxConcurrentTools)
		if err != nil {
			return nil, err
		}
		msgJSON, err = encodeToolResults(results)
		if err != nil {
			return nil, err
		}
	}
}

// sendOne posts one message through the transport and parses the
// reply envelope.
func sendOne(ctx context.Context, transport chatTransport, msgJSON string, opts OptionalArgs) (*Reply, error) {
	stop := wireCancel(ctx, transport.Cancel)
	defer stop()

	raw, err := transport.SendMessage(msgJSON, "", opts)
	if err != nil {
		if cerr := ctx.Err(); cerr != nil {
			return nil, cerr
		}
		return nil, fmt.Errorf("litertlm: send: %w", err)
	}
	return parseReply(raw)
}

// buildOptionalArgs materializes a Conversation OptionalArgs handle
// from the resolved per-call cfg. Returns OptionalArgs(0) (the C-side
// default) when no per-call knob is set. Caller must Delete the
// returned handle.
func buildOptionalArgs(cfg runtimeConfig) (OptionalArgs, error) {
	if cfg.visualTokenBudget == nil {
		return 0, nil
	}
	o, err := NewOptionalArgs()
	if err != nil {
		return 0, fmt.Errorf("litertlm: per-call options: %w", err)
	}
	o.SetVisualTokenBudget(*cfg.visualTokenBudget)
	return o, nil
}

// allCallsDispatchable reports whether every tool call in reply maps
// to a ManagedTool in the chat's registry. Returns false (so the
// dispatcher bails to manual mode) when any call is unknown or maps
// to a RawTool.
func (ch *Chat) allCallsDispatchable(reply *Reply) bool {
	if len(ch.tools) == 0 {
		return false
	}
	for _, call := range reply.ToolCalls() {
		def, ok := ch.tools[call.Function.Name]
		if !ok {
			return false
		}
		if _, ok := def.(dispatchable); !ok {
			return false
		}
	}
	return true
}

// invokeAll dispatches every call and returns the results bundled for
// one tool-role message. maxConcurrent <= 1 runs sequentially (the
// default); maxConcurrent > 1 runs handlers in parallel capped at
// that many in-flight. Result ordering matches call order regardless
// of completion order.
//
// A handler error obeys the tool's ToolPolicy: ToolPolicyReturnOnError
// propagates as a Go error and stops the loop; ToolPolicyInformOnError
// marshals the error message as the tool's response so the model can
// react. The first ReturnOnError failure in call order wins.
func (ch *Chat) invokeAll(ctx context.Context, calls []ToolCall, maxConcurrent int) ([]toolResult, error) {
	return runDispatch(ctx, calls, maxConcurrent, ch.dispatchManaged)
}

// invokeOrInform is the streaming-path dispatcher. Dispatchable calls
// invoke their handler (with the per-tool ToolPolicy honored); calls
// that name an unknown tool, or that name a RawTool which has no
// auto-handler in the streaming context, are answered with a
// synthesized error tool result. The model receives the inform-back
// in the next turn and can recover (e.g. by apologizing or by
// choosing a real tool). The dispatch loop's hop cap still applies,
// so a model that hallucinates indefinitely is bounded.
//
// maxConcurrent has the same semantics as invokeAll.
func (ch *Chat) invokeOrInform(ctx context.Context, calls []ToolCall, maxConcurrent int) ([]toolResult, error) {
	return runDispatch(ctx, calls, maxConcurrent, ch.dispatchOrInform)
}

// dispatchManaged is the per-call entry for invokeAll. The caller has
// already verified every call maps to a dispatchable ManagedTool.
func (ch *Chat) dispatchManaged(ctx context.Context, call ToolCall) (toolResult, error) {
	d := ch.tools[call.Function.Name].(dispatchable)
	argsJSON, err := json.Marshal(call.Function.Arguments)
	if err != nil {
		return toolResult{}, fmt.Errorf("litertlm: tool %q: marshal args: %w", call.Function.Name, err)
	}
	out, err := d.invoke(ctx, argsJSON)
	if err != nil {
		if d.policy() == ToolPolicyInformOnError {
			return toolResult{
				name:     call.Function.Name,
				response: map[string]any{"error": err.Error()},
			}, nil
		}
		return toolResult{}, fmt.Errorf("litertlm: tool %q: %w", call.Function.Name, err)
	}
	return toolResult{name: call.Function.Name, response: out}, nil
}

// dispatchOrInform is the per-call entry for invokeOrInform. Unknown
// tools and non-dispatchable tools (RawTool) return a synthesized
// inform-back result rather than failing the batch.
func (ch *Chat) dispatchOrInform(ctx context.Context, call ToolCall) (toolResult, error) {
	def, ok := ch.tools[call.Function.Name]
	if !ok {
		return toolResult{
			name: call.Function.Name,
			response: map[string]any{
				"error":           fmt.Sprintf("tool %q is not registered", call.Function.Name),
				"available_tools": ch.dispatchableToolNames(),
			},
		}, nil
	}
	d, isD := def.(dispatchable)
	if !isD {
		return toolResult{
			name: call.Function.Name,
			response: map[string]any{
				"error": fmt.Sprintf("tool %q is not auto-dispatchable in a streaming turn", call.Function.Name),
			},
		}, nil
	}
	argsJSON, err := json.Marshal(call.Function.Arguments)
	if err != nil {
		return toolResult{}, fmt.Errorf("litertlm: tool %q: marshal args: %w", call.Function.Name, err)
	}
	out, err := d.invoke(ctx, argsJSON)
	if err != nil {
		if d.policy() == ToolPolicyInformOnError {
			return toolResult{
				name:     call.Function.Name,
				response: map[string]any{"error": err.Error()},
			}, nil
		}
		return toolResult{}, fmt.Errorf("litertlm: tool %q: %w", call.Function.Name, err)
	}
	return toolResult{name: call.Function.Name, response: out}, nil
}

// runDispatch invokes perCall for every call, sequentially when
// maxConcurrent <= 1 and in parallel capped at maxConcurrent otherwise.
// Results preserve call-order regardless of completion order. The first
// handler to fail (in real time) wins; sibling handlers see ctx
// cancellation and may bail. Outer-ctx cancellation propagates as
// ctx.Err.
func runDispatch(ctx context.Context, calls []ToolCall, maxConcurrent int, perCall func(context.Context, ToolCall) (toolResult, error)) ([]toolResult, error) {
	if maxConcurrent <= 1 || len(calls) <= 1 {
		results := make([]toolResult, len(calls))
		for i, call := range calls {
			r, err := perCall(ctx, call)
			if err != nil {
				return nil, err
			}
			results[i] = r
		}
		return results, nil
	}

	subCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	results := make([]toolResult, len(calls))
	var (
		wg       sync.WaitGroup
		errOnce  sync.Once
		firstErr error
	)
	sem := make(chan struct{}, maxConcurrent)
	for i, call := range calls {
		wg.Add(1)
		go func(i int, call ToolCall) {
			defer wg.Done()
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-subCtx.Done():
				return
			}
			r, err := perCall(subCtx, call)
			if err != nil {
				errOnce.Do(func() {
					firstErr = err
					cancel()
				})
				return
			}
			results[i] = r
		}(i, call)
	}
	wg.Wait()

	if firstErr != nil {
		return nil, firstErr
	}
	if outerErr := ctx.Err(); outerErr != nil {
		return nil, outerErr
	}
	return results, nil
}

// dispatchableToolNames returns the sorted names of the chat's
// auto-dispatchable tools. Used by invokeOrInform to surface valid
// alternatives in an inform-back message.
func (ch *Chat) dispatchableToolNames() []string {
	names := make([]string, 0, len(ch.tools))
	for name, def := range ch.tools {
		if _, ok := def.(dispatchable); ok {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
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

// encodeMessages renders []Message as the JSON array the C side
// expects: each entry has {role, content:[<parts>]}. Empty Role is
// rejected; empty Parts emits an empty content array.
func encodeMessages(msgs []Message) (string, error) {
	if len(msgs) == 0 {
		return "", nil
	}
	list := make([]map[string]any, len(msgs))
	for i, m := range msgs {
		if m.Role == "" {
			return "", fmt.Errorf("litertlm: initial message %d: empty role", i)
		}
		list[i] = map[string]any{
			"role":    m.Role,
			"content": renderPartsContent(m.Parts),
		}
	}
	b, err := json.Marshal(list)
	if err != nil {
		return "", fmt.Errorf("marshal initial messages: %w", err)
	}
	return string(b), nil
}
