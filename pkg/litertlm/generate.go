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

// Generate runs synchronous inference for prompt and returns the assistant text.
func (c *Client) Generate(ctx context.Context, prompt string, opts ...RuntimeOption) (string, error) {
	return c.generateMulti(ctx, []Part{Text(prompt)}, resolveRuntimeConfig(opts))
}

// GenerateMulti is the multimodal sibling of Generate. parts may
// contain text, image, and audio Parts in any order; the model
// requires at least one text part to drive generation.
//
// The underlying C-side Conversation handle is created and destroyed
// for this single call.
func (c *Client) GenerateMulti(ctx context.Context, parts []Part, opts ...RuntimeOption) (string, error) {
	return c.generateMulti(ctx, parts, resolveRuntimeConfig(opts))
}

func (c *Client) generateMulti(ctx context.Context, parts []Part, cfg runtimeConfig) (string, error) {
	text, _, err := c.runConversation(ctx, parts, cfg)
	return text, err
}

// runConversation builds a temporary Conversation, sends the message,
// and returns the assistant's text plus a benchmark snapshot when the Client
// was constructed with WithBenchmarkEnabled.
func (c *Client) runConversation(ctx context.Context, parts []Part, cfg runtimeConfig) (string, *Benchmark, error) {
	conv, convCfg, err := c.openConversation(cfg)
	if err != nil {
		return "", nil, err
	}
	defer conv.Delete()
	defer convCfg.Delete()

	stop := wireCancel(ctx, conv.Cancel)
	defer stop()

	msgJSON, err := partsToConversationMessage(parts)
	if err != nil {
		return "", nil, err
	}

	optArgs, err := buildOptionalArgs(cfg)
	if err != nil {
		return "", nil, err
	}
	defer optArgs.Delete()

	raw, err := conv.SendMessage(msgJSON, "", optArgs)
	if err != nil {
		if cerr := ctx.Err(); cerr != nil {
			return "", nil, cerr
		}
		return "", nil, fmt.Errorf("litertlm: Generate: %w", err)
	}
	text, err := assistantText(raw)
	if err != nil {
		return "", nil, err
	}
	return text, captureConversationBenchmark(c, conv), nil
}

// openConversation creates a fresh ConversationConfig + Conversation
// pair with cfg's per-call SessionConfig (sampler, max-output-tokens) attached.
// Caller must Delete both handles.
func (c *Client) openConversation(cfg runtimeConfig) (Conversation, ConversationConfig, error) {
	sampler := cfg.sampler
	if sampler == nil {
		sampler = c.cfg.defaultSampler
	}

	var sessCfg SessionConfig
	if cfg.maxOutputTokens > 0 || sampler != nil {
		var err error
		sessCfg, err = NewSessionConfig()
		if err != nil {
			return 0, 0, err
		}
		if cfg.maxOutputTokens > 0 {
			if err = sessCfg.SetMaxOutputTokens(cfg.maxOutputTokens); err != nil {
				sessCfg.Delete()
				return 0, 0, err
			}
		}
		if sampler != nil {
			if err = sessCfg.SetSamplerParams(*sampler); err != nil {
				sessCfg.Delete()
				return 0, 0, err
			}
		}
	}

	engine, err := c.openEngine()
	if err != nil {
		sessCfg.Delete()
		return 0, 0, err
	}
	constrained := cfg.constraintType != ConstraintNone
	convCfg, err := NewConversationConfig(engine, sessCfg, "", "", "", constrained)
	sessCfg.Delete()
	if err != nil {
		return 0, 0, err
	}

	conv, err := engine.NewConversation(convCfg)
	if err != nil {
		convCfg.Delete()
		return 0, 0, err
	}
	return conv, convCfg, nil
}

// GenerateStream returns an iterator over response chunks. Cancelling
// ctx aborts the in-flight stream via Conversation.Cancel.
func (c *Client) GenerateStream(ctx context.Context, prompt string, opts ...RuntimeOption) iter.Seq2[Chunk, error] {
	return c.generateMultiStream(ctx, []Part{Text(prompt)}, resolveRuntimeConfig(opts))
}

// GenerateMultiStream is the multimodal sibling of GenerateStream.
// See GenerateMulti for parts semantics.
func (c *Client) GenerateMultiStream(ctx context.Context, parts []Part, opts ...RuntimeOption) iter.Seq2[Chunk, error] {
	return c.generateMultiStream(ctx, parts, resolveRuntimeConfig(opts))
}

func (c *Client) generateMultiStream(ctx context.Context, parts []Part, cfg runtimeConfig) iter.Seq2[Chunk, error] {
	return c.streamConversation(ctx, parts, cfg)
}

// streamConversation drives a Conversation streaming send. On early break, it
// cancels the native stream and drains remaining chunks so conv.Delete() never
// races an active C++ callback.
func (c *Client) streamConversation(ctx context.Context, parts []Part, cfg runtimeConfig) iter.Seq2[Chunk, error] {
	return func(yield func(Chunk, error) bool) {
		conv, convCfg, err := c.openConversation(cfg)
		if err != nil {
			yield(Chunk{}, err)
			return
		}
		defer conv.Delete()
		defer convCfg.Delete()

		stop := wireCancel(ctx, conv.Cancel)
		defer stop()

		msgJSON, err := partsToConversationMessage(parts)
		if err != nil {
			yield(Chunk{}, err)
			return
		}

		optArgs, err := buildOptionalArgs(cfg)
		if err != nil {
			yield(Chunk{}, err)
			return
		}
		defer optArgs.Delete()

		ch := conv.sendMessageStreamCh(msgJSON, "", optArgs, c.singleThreaded())
		broken := false
		for sc := range ch {
			if broken {
				continue
			}
			chOut := Chunk{Text: extractStreamChunkText(sc.Text), Final: sc.Final}
			if !yield(chOut, sc.Err) {
				broken = true
				conv.Cancel()
				continue
			}
			if sc.Err != nil {
				broken = true
			}
		}
	}
}

// GenerateResponse is the rich-output sibling of Generate, wrapping the
// output text and optional captured benchmark metrics.
func (c *Client) GenerateResponse(ctx context.Context, prompt string, opts ...RuntimeOption) (*Response, error) {
	return c.generateMultiResponse(ctx, []Part{Text(prompt)}, resolveRuntimeConfig(opts))
}

// GenerateMultiResponse is the multimodal sibling of GenerateResponse.
// See GenerateMulti for parts semantics.
func (c *Client) GenerateMultiResponse(ctx context.Context, parts []Part, opts ...RuntimeOption) (*Response, error) {
	return c.generateMultiResponse(ctx, parts, resolveRuntimeConfig(opts))
}

func (c *Client) generateMultiResponse(ctx context.Context, parts []Part, cfg runtimeConfig) (*Response, error) {
	text, bench, err := c.runConversation(ctx, parts, cfg)
	if err != nil {
		return nil, err
	}
	resp := newTextResponse(text)
	resp.bench = bench
	return resp, nil
}

// captureConversationBenchmark snapshots conv's BenchmarkInfo into a
// *Benchmark when c was constructed with WithBenchmarkEnabled.
// Returns nil otherwise.
func captureConversationBenchmark(c *Client, conv Conversation) *Benchmark {
	if c.cfg.benchmarkEnabled == nil || !*c.cfg.benchmarkEnabled {
		return nil
	}
	bi, err := conv.BenchmarkInfo()
	if err != nil {
		return nil
	}
	defer bi.Delete()
	return snapshotBenchmark(bi)
}

// wireCancel arranges for ctx cancellation to call cancelFn (typically
// Conversation.Cancel). Returns a stop function the caller defers to
// release the watcher goroutine. No-op when ctx cannot be cancelled.
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
