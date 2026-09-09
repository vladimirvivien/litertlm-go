package litertlm

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sync"
)

const defaultMaxToolHops = 5

type chatTransport interface {
	SendMessage(messageJSON, extraContext string, opts OptionalArgs) (string, error)
	SendMessageStreamCh(messageJSON, extraContext string, opts OptionalArgs) <-chan StreamChunk
	Cancel()
}

var _ chatTransport = Conversation(0)

// ErrToolHopsExceeded matches errors returned when Chat.Send exceeds
// the dispatch hop cap. Use errors.As to a *ToolHopsError to inspect
// the partial last reply.
var ErrToolHopsExceeded = errors.New("litertlm: tool hops exceeded")

// ToolHopsError carries the last assistant reply and the cap that was reached.
type ToolHopsError struct {
	LastReply *Reply
	Hops      int
}

func (e *ToolHopsError) Error() string {
	return fmt.Sprintf("litertlm: tool hops exceeded (cap %d)", e.Hops)
}

func (e *ToolHopsError) Unwrap() error { return ErrToolHopsExceeded }

func (ch *Chat) effectiveMaxHops() int {
	return cmp.Or(ch.maxToolHops, defaultMaxToolHops)
}

func (ch *Chat) streamWithDispatch(ctx context.Context, transport chatTransport, msgJSON string, cfg runtimeConfig, yield func(Chunk, error) bool) {
	optArgs, err := buildOptionalArgs(cfg)
	if err != nil {
		yield(Chunk{}, err)
		return
	}
	defer optArgs.Delete()

	maxHops := ch.effectiveMaxHops()

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
			yield(Chunk{Text: "", Final: true}, nil)
			return
		}

		reply := &Reply{toolCalls: calls}
		if hop >= maxHops {
			yield(Chunk{}, &ToolHopsError{LastReply: reply, Hops: maxHops})
			return
		}
		results, dispatchErr := ch.invokeOrInform(ctx, calls, cfg.maxConcurrentTools)
		if dispatchErr != nil {
			yield(Chunk{}, dispatchErr)
			return
		}
		msgJSON, err = encodeToolResults(results)
		if err != nil {
			yield(Chunk{}, err)
			return
		}
	}
}

func (ch *Chat) streamOne(ctx context.Context, transport chatTransport, msgJSON string, opts OptionalArgs, calls *[]ToolCall, yield func(Chunk, error) bool) (bool, error) {
	stop := wireCancel(ctx, transport.Cancel)
	defer stop()

	chOut := transport.SendMessageStreamCh(msgJSON, "", opts)
	cancelled := false
	var streamErr error

	for sc := range chOut {
		if sc.Err != nil {
			if !cancelled {
				streamErr = sc.Err
			}
			cancelled = true
			continue
		}
		if cancelled {
			continue
		}
		text, chunkCalls := extractStreamEnvelope(sc.Text)
		if len(chunkCalls) > 0 {
			*calls = append(*calls, chunkCalls...)
		}
		if text == "" {
			continue
		}
		if !yield(Chunk{Text: text, Final: false}, nil) {
			cancelled = true
			transport.Cancel()
			continue
		}
	}
	if streamErr != nil {
		return false, streamErr
	}
	return cancelled, nil
}

func (ch *Chat) send(ctx context.Context, transport chatTransport, msgJSON string, cfg runtimeConfig) (*Reply, error) {
	optArgs, err := buildOptionalArgs(cfg)
	if err != nil {
		return nil, err
	}
	defer optArgs.Delete()

	maxHops := ch.effectiveMaxHops()

	for hop := 0; ; hop++ {
		reply, err := sendOne(ctx, transport, msgJSON, optArgs)
		if err != nil {
			return nil, err
		}
		if !reply.HasToolCalls() || !ch.allCallsDispatchable(reply) || cfg.returnToolRequests {
			return reply, nil
		}
		if hop >= maxHops {
			return nil, &ToolHopsError{LastReply: reply, Hops: maxHops}
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

func (ch *Chat) invokeAll(ctx context.Context, calls []ToolCall, maxConcurrent int) ([]toolResult, error) {
	return runDispatch(ctx, calls, maxConcurrent, ch.dispatchManaged)
}

func (ch *Chat) invokeOrInform(ctx context.Context, calls []ToolCall, maxConcurrent int) ([]toolResult, error) {
	return runDispatch(ctx, calls, maxConcurrent, ch.dispatchOrInform)
}

func (ch *Chat) dispatchManaged(ctx context.Context, call ToolCall) (toolResult, error) {
	return invokeDispatchable(ctx, ch.tools[call.Function.Name].(dispatchable), call)
}

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
	return invokeDispatchable(ctx, d, call)
}

func invokeDispatchable(ctx context.Context, d dispatchable, call ToolCall) (toolResult, error) {
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
		wg.Go(func() {
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
		})
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

func (ch *Chat) dispatchableToolNames() []string {
	names := make([]string, 0, len(ch.tools))
	for name, def := range ch.tools {
		if _, ok := def.(dispatchable); ok {
			names = append(names, name)
		}
	}
	slices.Sort(names)
	return names
}
