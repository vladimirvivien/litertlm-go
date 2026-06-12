// Package litertgo adapts the pure-Go litert-go LLM runtime
// (github.com/vladimirvivien/litert-go/lm) to litertlm-go's Backend
// seam. A Client constructed with
// litertlm.WithEngineBackend(litertgo.Open(...)) serves the same
// Generate / Chat API without loading the C++ LiteRT-LM library.
//
// Supported surface: text generation (sync and streaming), multi-turn
// chat with a system prompt, sampler and max-output-token controls,
// and Tokenize. Tools, constrained decoding, extra context, initial
// messages, multimodal parts, scoring, cloning, rendering, token
// counting, and benchmarks are not yet implemented and return
// ErrUnsupported.
package litertgo

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

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
		ctx:    ctx,
		cancel: cancel,
	}, nil
}

func (b *Backend) NewChatTransport(s litertlm.ConversationSetup) (litertlm.ChatTransport, error) {
	if s.ToolsJSON != "" {
		return nil, fmt.Errorf("%w: tools", ErrUnsupported)
	}
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
	conv, err := b.engine.NewConversation(genOptions(s.MaxOutputTokens, s.Sampler, system))
	if err != nil {
		return nil, fmt.Errorf("litertgo: %w", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	return &transport{conv: conv, ctx: ctx, cancel: cancel}, nil
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

// session implements litertlm.SessionBackend. Mirroring the C++
// session, Generate runs the raw prompt — no chat-template wrapping.
type session struct {
	engine *lm.Engine
	opts   lm.GenOptions
	ctx    context.Context
	cancel context.CancelFunc
}

func (s *session) Generate(parts []litertlm.Part) ([]litertlm.Candidate, error) {
	prompt, err := partsText(parts)
	if err != nil {
		return nil, err
	}
	text, err := s.engine.Generate(s.ctx, prompt, false, s.opts)
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
		_, err = s.engine.GenerateStream(s.ctx, prompt, false, s.opts, func(piece string) {
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
// embedding-input models).
type transport struct {
	conv   lm.Conversation
	ctx    context.Context
	cancel context.CancelFunc
}

func (t *transport) SendMessage(messageJSON, extraContext string, args litertlm.RuntimeArgs) (string, error) {
	if extraContext != "" {
		return "", fmt.Errorf("%w: extra context", ErrUnsupported)
	}
	if args.VisualTokenBudget > 0 {
		return "", fmt.Errorf("%w: visual token budget", ErrUnsupported)
	}
	text, err := messageText(messageJSON)
	if err != nil {
		return "", err
	}
	reply, err := t.conv.Send(t.ctx, text)
	if err != nil {
		return "", err
	}
	return assistantEnvelope(reply)
}

func (t *transport) SendMessageStreamCh(messageJSON, extraContext string, args litertlm.RuntimeArgs) <-chan litertlm.StreamChunk {
	out := make(chan litertlm.StreamChunk)
	go func() {
		defer close(out)
		if extraContext != "" {
			t.emit(out, litertlm.StreamChunk{Err: fmt.Errorf("%w: extra context", ErrUnsupported), Final: true})
			return
		}
		if args.VisualTokenBudget > 0 {
			t.emit(out, litertlm.StreamChunk{Err: fmt.Errorf("%w: visual token budget", ErrUnsupported), Final: true})
			return
		}
		text, err := messageText(messageJSON)
		if err != nil {
			t.emit(out, litertlm.StreamChunk{Err: err, Final: true})
			return
		}
		// Pieces are emitted as plain text: the Chat dispatch loop's
		// envelope extraction passes non-JSON chunks through as text.
		_, err = t.conv.SendStream(t.ctx, text, func(piece string) {
			t.emit(out, litertlm.StreamChunk{Text: piece})
		})
		if err != nil {
			t.emit(out, litertlm.StreamChunk{Err: err, Final: true})
			return
		}
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

// messageText extracts the user text from one litertlm message
// envelope: {"role":"user","content":"..."} or the content-array
// form with text-typed items.
func messageText(messageJSON string) (string, error) {
	var msg struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal([]byte(messageJSON), &msg); err != nil {
		return "", fmt.Errorf("litertgo: parse message: %w", err)
	}
	if msg.Role != "" && msg.Role != "user" {
		return "", fmt.Errorf("%w: %q messages", ErrUnsupported, msg.Role)
	}

	var s string
	if err := json.Unmarshal(msg.Content, &s); err == nil {
		return s, nil
	}
	var items []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(msg.Content, &items); err == nil {
		var b strings.Builder
		for _, it := range items {
			if it.Type != "text" {
				return "", fmt.Errorf("%w: %q content parts", ErrUnsupported, it.Type)
			}
			b.WriteString(it.Text)
		}
		return b.String(), nil
	}
	return "", fmt.Errorf("litertgo: unrecognized message content: %s", msg.Content)
}

// assistantEnvelope wraps reply text in the assistant envelope the
// litertlm reply parser expects:
// {"role":"assistant","content":[{"type":"text","text":"..."}]}.
func assistantEnvelope(text string) (string, error) {
	b, err := json.Marshal(map[string]any{
		"role":    "assistant",
		"content": []map[string]string{{"type": "text", "text": text}},
	})
	if err != nil {
		return "", fmt.Errorf("litertgo: marshal reply envelope: %w", err)
	}
	return string(b), nil
}
