// Package litertgo adapts the pure-Go litert-go LLM runtime
// (github.com/vladimirvivien/litert-go/lm) to litertlm-go's Backend
// seam. A Client constructed with
// litertlm.WithEngineBackend(litertgo.Open(...)) serves the same
// Generate / Chat API without loading the C++ LiteRT-LM library.
//
// Supported surface: text generation (sync and streaming), multi-turn
// chat with a system prompt, function calling on tool-capable
// families (Gemma 4), single-image and single-audio first-turn
// messages (the litertlm multimodal Generate path; audio as 16 kHz
// mono WAV), sampler and max-output-token controls, and Tokenize.
// Constrained decoding, extra context, initial messages, multimodal
// parts mid-conversation, scoring, cloning, rendering, token
// counting, and benchmarks are not yet implemented and return
// ErrUnsupported.
package litertgo

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/vladimirvivien/litert-go/audio"
	"github.com/vladimirvivien/litert-go/lm"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

// ErrUnsupported marks seam capabilities the litert-go engine does not
// implement yet. Match with errors.Is.
var ErrUnsupported = errors.New("litertgo: not supported")

// Backend implements litertlm.Backend over a litert-go lm.Engine.
type Backend struct {
	engine *lm.Engine
}

var _ litertlm.Backend = (*Backend)(nil)

// Open loads modelPath into a litert-go engine and wraps it as a
// litertlm Backend. opts pass through to lm.Open (accelerator, lib
// dir, runtime fetch, metrics).
func Open(ctx context.Context, modelPath string, opts ...lm.Option) (*Backend, error) {
	engine, err := lm.Open(ctx, modelPath, opts...)
	if err != nil {
		return nil, fmt.Errorf("litertgo: %w", err)
	}
	return &Backend{engine: engine}, nil
}

// Engine returns the underlying lm.Engine for advanced use. Its
// lifetime is owned by the Backend; do not call Close on it.
func (b *Backend) Engine() *lm.Engine { return b.engine }

func (b *Backend) Tokenize(text string) ([]int32, error) {
	return b.engine.Tokenize(text)
}

func (b *Backend) NewSessionBackend(s litertlm.SessionSetup) (litertlm.SessionBackend, error) {
	ctx, cancel := context.WithCancel(context.Background())
	return &session{
		engine: b.engine,
		opts:   genOptions(s.MaxOutputTokens, s.Sampler, ""),
		// The C++ session renders prompts through the model's chat
		// template when one exists; match it, falling back to raw
		// completion for template-less models.
		chat:   b.engine.HasChatTemplate(),
		ctx:    ctx,
		cancel: cancel,
	}, nil
}

func (b *Backend) NewChatTransport(s litertlm.ConversationSetup) (litertlm.ChatTransport, error) {
	if s.MessagesJSON != "" {
		return nil, fmt.Errorf("%w: initial messages", ErrUnsupported)
	}
	if s.ConstrainedDecoding {
		return nil, fmt.Errorf("%w: constrained decoding", ErrUnsupported)
	}
	if s.ExtraContextJSON != "" {
		return nil, fmt.Errorf("%w: extra context", ErrUnsupported)
	}
	if s.FilterChannelContentFromKVCache != nil {
		return nil, fmt.Errorf("%w: KV-cache channel filtering", ErrUnsupported)
	}

	system, err := systemText(s.SystemMessageJSON)
	if err != nil {
		return nil, err
	}
	opts := genOptions(s.MaxOutputTokens, s.Sampler, system)
	opts.ToolsJSON = s.ToolsJSON
	conv, err := b.engine.NewConversation(opts)
	if err != nil {
		return nil, fmt.Errorf("litertgo: %w", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	return &transport{conv: conv, engine: b.engine, opts: opts, hasTools: s.ToolsJSON != "", ctx: ctx, cancel: cancel}, nil
}

func (b *Backend) Close() {
	b.engine.Close()
}

// defaultMaxOutputTokens caps generation when the caller sets no
// per-call limit. lm.GenOptions.MaxTokens is a hard loop bound (zero
// generates nothing), so an unset seam value needs a real cap;
// decoding still stops at EOS well before it on normal replies.
const defaultMaxOutputTokens = 4096

// genOptions maps the seam's Go-native session knobs onto lm.GenOptions.
// A nil sampler and SamplerGreedy both resolve to greedy decoding
// (lm treats Temp 0 as greedy).
func genOptions(maxOutputTokens int, p *litertlm.SamplerParams, system string) lm.GenOptions {
	if maxOutputTokens <= 0 {
		maxOutputTokens = defaultMaxOutputTokens
	}
	o := lm.GenOptions{MaxTokens: maxOutputTokens, System: system}
	if p == nil || p.Type == litertlm.SamplerGreedy {
		return o
	}
	o.Temp = p.Temperature
	o.TopK = int(p.TopK)
	o.Seed = int64(p.Seed)
	if p.Type == litertlm.SamplerTopP {
		o.TopP = p.TopP
	}
	return o
}

// session implements litertlm.SessionBackend.
type session struct {
	engine *lm.Engine
	opts   lm.GenOptions
	chat   bool
	ctx    context.Context
	cancel context.CancelFunc
}

func (s *session) Generate(parts []litertlm.Part) ([]litertlm.Candidate, error) {
	prompt, err := partsText(parts)
	if err != nil {
		return nil, err
	}
	text, err := s.engine.Generate(s.ctx, prompt, s.chat, s.opts)
	if err != nil {
		return nil, err
	}
	return []litertlm.Candidate{{Text: text}}, nil
}

func (s *session) GenerateStreamCh(parts []litertlm.Part) <-chan litertlm.StreamChunk {
	out := make(chan litertlm.StreamChunk)
	go func() {
		defer close(out)
		prompt, err := partsText(parts)
		if err != nil {
			s.emit(out, litertlm.StreamChunk{Err: err, Final: true})
			return
		}
		_, err = s.engine.GenerateStream(s.ctx, prompt, s.chat, s.opts, func(piece string) {
			s.emit(out, litertlm.StreamChunk{Text: piece})
		})
		if err != nil {
			s.emit(out, litertlm.StreamChunk{Err: err, Final: true})
			return
		}
		s.emit(out, litertlm.StreamChunk{Final: true})
	}()
	return out
}

// emit sends one chunk unless the session is cancelled or closed —
// the release path for a consumer that stops ranging the channel.
func (s *session) emit(out chan<- litertlm.StreamChunk, sc litertlm.StreamChunk) {
	select {
	case out <- sc:
	case <-s.ctx.Done():
	}
}

func (s *session) ScoreTexts(targets []string, storeTokenLengths bool) ([]litertlm.Candidate, error) {
	return nil, fmt.Errorf("%w: ScoreTexts", ErrUnsupported)
}

func (s *session) Benchmark() *litertlm.Benchmark { return nil }

func (s *session) Cancel() { s.cancel() }

func (s *session) Close() { s.cancel() }

// transport implements litertlm.ChatTransport over an lm.Conversation
// (KV-cache-reusing for token-input models, embedding session for
// embedding-input models). A first-turn image or audio message runs
// as an engine one-shot instead — lm's multimodal path is one-shot —
// which consumes the transport for further sends. That mirrors how
// the litertlm multimodal Generate path uses a transport: one fresh
// transport, one send, Close.
type transport struct {
	conv     lm.Conversation
	engine   *lm.Engine
	opts     lm.GenOptions
	hasTools bool
	turns    int
	consumed bool
	ctx      context.Context
	cancel   context.CancelFunc
}

func (t *transport) SendMessage(messageJSON, extraContext string, args litertlm.RuntimeArgs) (string, error) {
	if extraContext != "" {
		return "", fmt.Errorf("%w: extra context", ErrUnsupported)
	}
	if t.consumed {
		return "", fmt.Errorf("%w: send after a one-shot multimodal turn", ErrUnsupported)
	}
	msg, err := decodeMessage(messageJSON)
	if err != nil {
		return "", err
	}

	var reply string
	switch {
	case msg.Role == "tool":
		sender, ok := t.conv.(lm.ToolSender)
		if !ok {
			return "", fmt.Errorf("%w: tool results on %T", ErrUnsupported, t.conv)
		}
		reply, err = sender.SendToolResults(t.ctx, msg.Results)
	case len(msg.binary()) > 0:
		reply, err = t.sendModal(msg, args)
	default:
		reply, err = t.conv.Send(t.ctx, msg.text())
	}
	if err != nil {
		return "", err
	}
	t.turns++

	text, calls, err := t.engine.ExtractToolCalls(reply)
	if err != nil {
		return "", err
	}
	return replyEnvelope(text, calls)
}

// sendModal runs a first-turn image or audio message as an engine
// one-shot: the content items become a prompt with the modality
// marker at the binary item's position.
func (t *transport) sendModal(msg message, args litertlm.RuntimeArgs) (string, error) {
	if t.turns > 0 {
		return "", fmt.Errorf("%w: image/audio parts after the first turn", ErrUnsupported)
	}
	if t.hasTools {
		return "", fmt.Errorf("%w: tools with image/audio parts", ErrUnsupported)
	}
	binary := msg.binary()
	if len(binary) != 1 {
		return "", fmt.Errorf("%w: %d image/audio parts in one message (max 1)", ErrUnsupported, len(binary))
	}

	var prompt strings.Builder
	for _, it := range msg.Items {
		switch it.kind {
		case "text":
			prompt.WriteString(it.text)
		case "image":
			prompt.WriteString("<start_of_image>")
		case "audio":
			prompt.WriteString("<start_of_audio>")
		}
	}
	t.consumed = true

	switch binary[0].kind {
	case "image":
		return t.engine.GenerateFromImage(t.ctx, prompt.String(), binary[0].data, args.VisualTokenBudget, t.opts)
	default:
		pcm, err := audio.DecodeWAV(binary[0].data)
		if err != nil {
			return "", fmt.Errorf("litertgo: %w", err)
		}
		return t.engine.GenerateFromAudio(t.ctx, prompt.String(), pcm, t.opts)
	}
}

func (t *transport) SendMessageStreamCh(messageJSON, extraContext string, args litertlm.RuntimeArgs) <-chan litertlm.StreamChunk {
	msg, err := decodeMessage(messageJSON)
	if err != nil {
		out := make(chan litertlm.StreamChunk, 1)
		out <- litertlm.StreamChunk{Err: err, Final: true}
		close(out)
		return out
	}

	// Tool turns and multimodal turns must arrive as one parsed
	// envelope (mirroring the C++ stream, whose chunks are
	// envelopes); buffer the turn and emit it whole.
	if t.hasTools || len(msg.binary()) > 0 {
		out := make(chan litertlm.StreamChunk, 2)
		go func() {
			defer close(out)
			env, err := t.SendMessage(messageJSON, extraContext, args)
			if err != nil {
				t.emit(out, litertlm.StreamChunk{Err: err, Final: true})
				return
			}
			t.emit(out, litertlm.StreamChunk{Text: env})
			t.emit(out, litertlm.StreamChunk{Final: true})
		}()
		return out
	}

	out := make(chan litertlm.StreamChunk)
	go func() {
		defer close(out)
		if extraContext != "" {
			t.emit(out, litertlm.StreamChunk{Err: fmt.Errorf("%w: extra context", ErrUnsupported), Final: true})
			return
		}
		if t.consumed {
			t.emit(out, litertlm.StreamChunk{Err: fmt.Errorf("%w: send after a one-shot multimodal turn", ErrUnsupported), Final: true})
			return
		}
		if msg.Role == "tool" {
			t.emit(out, litertlm.StreamChunk{Err: fmt.Errorf("%w: tool results without tools configured", ErrUnsupported), Final: true})
			return
		}
		// Pieces are emitted as plain text: the Chat dispatch loop's
		// envelope extraction passes non-JSON chunks through as text.
		_, err = t.conv.SendStream(t.ctx, msg.text(), func(piece string) {
			t.emit(out, litertlm.StreamChunk{Text: piece})
		})
		if err != nil {
			t.emit(out, litertlm.StreamChunk{Err: err, Final: true})
			return
		}
		t.turns++
		t.emit(out, litertlm.StreamChunk{Final: true})
	}()
	return out
}

func (t *transport) emit(out chan<- litertlm.StreamChunk, sc litertlm.StreamChunk) {
	select {
	case out <- sc:
	case <-t.ctx.Done():
	}
}

func (t *transport) Render(messageJSON string) (string, error) {
	return "", fmt.Errorf("%w: Render", ErrUnsupported)
}

func (t *transport) TokenCount() (int, error) {
	return 0, fmt.Errorf("%w: TokenCount", ErrUnsupported)
}

func (t *transport) Clone() (litertlm.ChatTransport, error) {
	return nil, fmt.Errorf("%w: Clone", ErrUnsupported)
}

func (t *transport) Benchmark() *litertlm.Benchmark { return nil }

func (t *transport) Cancel() { t.cancel() }

func (t *transport) Close() {
	t.cancel()
	t.conv.Close()
}

// partsText flattens text parts into one prompt string. Image and
// audio parts are not routed through this backend yet.
func partsText(parts []litertlm.Part) (string, error) {
	var b strings.Builder
	for _, p := range parts {
		if !p.IsText() {
			return "", fmt.Errorf("%w: image/audio parts", ErrUnsupported)
		}
		b.WriteString(p.TextContent())
	}
	return b.String(), nil
}

// systemText decodes the seam's system-message JSON (a JSON-encoded
// string) into plain text. Empty input means no system prompt.
func systemText(systemJSON string) (string, error) {
	if systemJSON == "" {
		return "", nil
	}
	var s string
	if err := json.Unmarshal([]byte(systemJSON), &s); err != nil {
		return "", fmt.Errorf("litertgo: parse system message: %w", err)
	}
	return s, nil
}

// contentItem is one entry of a user message's content array, in
// message order.
type contentItem struct {
	kind string // "text", "image", "audio"
	text string
	data []byte
}

// message is one decoded litertlm message envelope: user content
// items, or tool results for role "tool".
type message struct {
	Role    string
	Items   []contentItem
	Results []lm.ToolResult
}

// text concatenates the message's text items.
func (m message) text() string {
	var b strings.Builder
	for _, it := range m.Items {
		if it.kind == "text" {
			b.WriteString(it.text)
		}
	}
	return b.String()
}

// binary returns the message's image/audio items.
func (m message) binary() []contentItem {
	var out []contentItem
	for _, it := range m.Items {
		if it.kind != "text" {
			out = append(out, it)
		}
	}
	return out
}

// decodeMessage parses a litertlm message envelope:
// {"role":"user","content":"..."} (or the content-array form with
// text / base64-blob image / base64-blob audio items), or the
// tool-result form {"role":"tool","content":[{"name":...,"response":...}]}.
func decodeMessage(messageJSON string) (message, error) {
	var msg struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal([]byte(messageJSON), &msg); err != nil {
		return message{}, fmt.Errorf("litertgo: parse message: %w", err)
	}

	switch msg.Role {
	case "tool":
		var items []struct {
			Name     string          `json:"name"`
			Response json.RawMessage `json:"response"`
		}
		if err := json.Unmarshal(msg.Content, &items); err != nil {
			return message{}, fmt.Errorf("litertgo: parse tool results: %w", err)
		}
		results := make([]lm.ToolResult, len(items))
		for i, it := range items {
			var resp any
			if err := json.Unmarshal(it.Response, &resp); err != nil {
				return message{}, fmt.Errorf("litertgo: parse tool response %q: %w", it.Name, err)
			}
			results[i] = lm.ToolResult{Name: it.Name, Response: resp}
		}
		return message{Role: "tool", Results: results}, nil

	case "", "user":
		var s string
		if err := json.Unmarshal(msg.Content, &s); err == nil {
			return message{Role: "user", Items: []contentItem{{kind: "text", text: s}}}, nil
		}
		var items []struct {
			Type string `json:"type"`
			Text string `json:"text"`
			Blob string `json:"blob"`
		}
		if err := json.Unmarshal(msg.Content, &items); err != nil {
			return message{}, fmt.Errorf("litertgo: unrecognized message content: %s", msg.Content)
		}
		out := message{Role: "user", Items: make([]contentItem, 0, len(items))}
		for _, it := range items {
			switch it.Type {
			case "text":
				out.Items = append(out.Items, contentItem{kind: "text", text: it.Text})
			case "image", "audio":
				data, err := base64.StdEncoding.DecodeString(it.Blob)
				if err != nil {
					return message{}, fmt.Errorf("litertgo: decode %s blob: %w", it.Type, err)
				}
				out.Items = append(out.Items, contentItem{kind: it.Type, data: data})
			default:
				return message{}, fmt.Errorf("%w: %q content parts", ErrUnsupported, it.Type)
			}
		}
		return out, nil

	default:
		return message{}, fmt.Errorf("%w: %q messages", ErrUnsupported, msg.Role)
	}
}

// replyEnvelope wraps a turn's text and tool calls in the assistant
// envelope the litertlm reply parser expects: content as a text-part
// array, tool calls as
// {"type":"function","function":{"name":...,"arguments":{...}}}.
func replyEnvelope(text string, calls []lm.ToolCall) (string, error) {
	msg := map[string]any{"role": "assistant"}
	if text != "" || len(calls) == 0 {
		msg["content"] = []map[string]string{{"type": "text", "text": text}}
	}
	if len(calls) > 0 {
		tcs := make([]map[string]any, len(calls))
		for i, c := range calls {
			tcs[i] = map[string]any{
				"type":     "function",
				"function": map[string]any{"name": c.Name, "arguments": c.Args},
			}
		}
		msg["tool_calls"] = tcs
	}
	b, err := json.Marshal(msg)
	if err != nil {
		return "", fmt.Errorf("litertgo: marshal reply envelope: %w", err)
	}
	return string(b), nil
}
