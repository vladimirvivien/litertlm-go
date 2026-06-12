package litertlm

import (
	"context"
	"fmt"
	"iter"
)

// Chunk is one piece of a streaming Generate response.
type Chunk struct {
	Text  string
	Final bool
}

// Generate runs synchronous inference for prompt and returns the first
// candidate's text. Cancelling ctx aborts the in-flight inference (via
// Session.Cancel) and returns ctx.Err.
func (c *Client) Generate(ctx context.Context, prompt string, opts ...RuntimeOption) (string, error) {
	return c.generateMulti(ctx, []Part{Text(prompt)}, resolveRuntimeConfig(opts))
}

// GenerateMulti is the multimodal sibling of Generate. parts may
// contain text, image, and audio Parts in any order; the model
// requires at least one text part to drive generation.
//
// Image and audio Parts require WithVisionBackend / WithAudioBackend
// at New time and a model whose .litertlm package includes the
// corresponding tower.
func (c *Client) GenerateMulti(ctx context.Context, parts []Part, opts ...RuntimeOption) (string, error) {
	return c.generateMulti(ctx, parts, resolveRuntimeConfig(opts))
}

// generateMulti is the resolved-config sibling of GenerateMulti.
// Generate, GenerateMulti, and GenerateData all funnel through here.
//
// Routing: text-only parts use Session.GenerateContent. Parts
// containing image or audio go through a Conversation, whose
// pipeline preprocesses the bytes before invoking the session.
func (c *Client) generateMulti(ctx context.Context, parts []Part, cfg runtimeConfig) (string, error) {
	if partsHasBinary(parts) {
		text, _, err := c.runMultimodalConversation(ctx, parts, cfg)
		return text, err
	}

	sess, err := c.openSession(cfg)
	if err != nil {
		return "", err
	}
	defer sess.Close()

	stop := wireCancel(ctx, sess.Cancel)
	defer stop()

	cands, err := sess.Generate(parts)
	if err != nil {
		if cerr := ctx.Err(); cerr != nil {
			return "", cerr
		}
		return "", err
	}
	if len(cands) == 0 {
		return "", fmt.Errorf("litertlm: Generate: no candidates returned")
	}
	return cands[0].Text, nil
}

// runMultimodalConversation builds a Conversation, sends the
// content-array message produced from parts, and returns the
// assistant's text plus a benchmark snapshot when the Client was
// constructed with WithBenchmarkEnabled. Used when parts contains
// image or audio Parts.
func (c *Client) runMultimodalConversation(ctx context.Context, parts []Part, cfg runtimeConfig) (string, *Benchmark, error) {
	transport, err := c.openMultimodalTransport(cfg)
	if err != nil {
		return "", nil, err
	}
	defer transport.Close()

	stop := wireCancel(ctx, transport.Cancel)
	defer stop()

	msgJSON, err := partsToConversationMessage(parts)
	if err != nil {
		return "", nil, err
	}

	raw, err := transport.SendMessage(msgJSON, "", RuntimeArgs{})
	if err != nil {
		if cerr := ctx.Err(); cerr != nil {
			return "", nil, cerr
		}
		return "", nil, fmt.Errorf("litertlm: GenerateMulti: %w", err)
	}
	text, err := assistantText(raw)
	if err != nil {
		return "", nil, err
	}
	return text, transport.Benchmark(), nil
}

// openMultimodalTransport opens a bare ChatTransport carrying cfg's
// per-call session knobs (sampler, max-output-tokens). Caller must
// Close the transport.
func (c *Client) openMultimodalTransport(cfg runtimeConfig) (ChatTransport, error) {
	sampler := cfg.sampler
	if sampler == nil {
		sampler = c.cfg.defaultSampler
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil, fmt.Errorf("litertlm: Client is closed")
	}
	return c.backend.NewChatTransport(ConversationSetup{
		Sampler:         sampler,
		MaxOutputTokens: cfg.maxOutputTokens,
	})
}

// GenerateStream returns an iterator over response chunks. Cancelling
// ctx aborts the stream via Session.Cancel; the iterator yields the
// surfaced error and closes.
//
//	for chunk, err := range client.GenerateStream(ctx, prompt) {
//	    if err != nil { return err }
//	    fmt.Print(chunk.Text)
//	}
func (c *Client) GenerateStream(ctx context.Context, prompt string, opts ...RuntimeOption) iter.Seq2[Chunk, error] {
	return c.generateMultiStream(ctx, []Part{Text(prompt)}, resolveRuntimeConfig(opts))
}

// GenerateMultiStream is the multimodal sibling of GenerateStream.
// See GenerateMulti for parts semantics.
func (c *Client) GenerateMultiStream(ctx context.Context, parts []Part, opts ...RuntimeOption) iter.Seq2[Chunk, error] {
	return c.generateMultiStream(ctx, parts, resolveRuntimeConfig(opts))
}

func (c *Client) generateMultiStream(ctx context.Context, parts []Part, cfg runtimeConfig) iter.Seq2[Chunk, error] {
	if partsHasBinary(parts) {
		return c.streamMultimodalConversation(ctx, parts, cfg)
	}
	return func(yield func(Chunk, error) bool) {
		sess, err := c.openSession(cfg)
		if err != nil {
			yield(Chunk{}, err)
			return
		}
		defer sess.Close()

		stop := wireCancel(ctx, sess.Cancel)
		defer stop()

		for sc := range sess.GenerateStreamCh(parts) {
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

// streamMultimodalConversation drives a Conversation streaming send
// for multimodal parts. Mirrors the Session-path streamer.
func (c *Client) streamMultimodalConversation(ctx context.Context, parts []Part, cfg runtimeConfig) iter.Seq2[Chunk, error] {
	return func(yield func(Chunk, error) bool) {
		transport, err := c.openMultimodalTransport(cfg)
		if err != nil {
			yield(Chunk{}, err)
			return
		}
		defer transport.Close()

		stop := wireCancel(ctx, transport.Cancel)
		defer stop()

		msgJSON, err := partsToConversationMessage(parts)
		if err != nil {
			yield(Chunk{}, err)
			return
		}

		for sc := range transport.SendMessageStreamCh(msgJSON, "", RuntimeArgs{}) {
			ch := Chunk{Text: extractStreamChunkText(sc.Text), Final: sc.Final}
			if !yield(ch, sc.Err) {
				return
			}
			if sc.Err != nil {
				return
			}
		}
	}
}

// GenerateResponse is the rich-output sibling of Generate. The
// returned *Response exposes per-candidate text plus score and
// token-length accessors. Lifetime is GC-managed via
// runtime.AddCleanup — see the Response godoc.
func (c *Client) GenerateResponse(ctx context.Context, prompt string, opts ...RuntimeOption) (*Response, error) {
	return c.generateMultiResponse(ctx, []Part{Text(prompt)}, resolveRuntimeConfig(opts))
}

// GenerateMultiResponse is the multimodal sibling of GenerateResponse.
// See GenerateMulti for parts semantics.
func (c *Client) GenerateMultiResponse(ctx context.Context, parts []Part, opts ...RuntimeOption) (*Response, error) {
	return c.generateMultiResponse(ctx, parts, resolveRuntimeConfig(opts))
}

func (c *Client) generateMultiResponse(ctx context.Context, parts []Part, cfg runtimeConfig) (*Response, error) {
	if partsHasBinary(parts) {
		text, bench, err := c.runMultimodalConversation(ctx, parts, cfg)
		if err != nil {
			return nil, err
		}
		resp := newTextResponse(text)
		resp.bench = bench
		return resp, nil
	}

	sess, err := c.openSession(cfg)
	if err != nil {
		return nil, err
	}
	defer sess.Close()

	stop := wireCancel(ctx, sess.Cancel)
	defer stop()

	cands, err := sess.Generate(parts)
	if err != nil {
		if cerr := ctx.Err(); cerr != nil {
			return nil, cerr
		}
		return nil, err
	}
	resp := newResponse(cands)
	resp.bench = sess.Benchmark()
	return resp, nil
}

// wireCancel arranges for ctx cancellation to call cancelFn (typically
// Session.Cancel or Conversation.Cancel). Returns a stop function the
// caller defers to release the watcher goroutine. No-op (returns a
// no-op stop) when ctx cannot be cancelled.
func wireCancel(ctx context.Context, cancelFn func()) (stop func()) {
	if ctx == nil || ctx.Done() == nil {
		return func() {}
	}
	done := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			cancelFn()
		case <-done:
		}
	}()
	return func() { close(done) }
}
