package litertlm

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// chunk envelopes for the streaming tests
func textChunk(s string, final bool) StreamChunk {
	return StreamChunk{Text: textReply(s), Final: final}
}

func toolCallChunk(name string, args map[string]any) StreamChunk {
	return StreamChunk{Text: toolCallReply(name, args), Final: false}
}

func TestStream_TextOnly(t *testing.T) {
	transport := &fakeTransport{
		streamReplies: [][]StreamChunk{{
			textChunk("hello", false),
			textChunk(" world", false),
			{Final: true},
		}},
	}
	ch := &Chat{conv: Conversation(1)}
	var got strings.Builder
	var sawFinal bool
	for chunk, err := range streamIter(ch, transport, `{"role":"user","content":"hi"}`) {
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		got.WriteString(chunk.Text)
		if chunk.Final {
			sawFinal = true
		}
	}
	if !sawFinal {
		t.Error("stream did not yield Final")
	}
	if got.String() != "hello world" {
		t.Errorf("text = %q, want %q", got.String(), "hello world")
	}
}

func TestStream_DispatchesManagedTool(t *testing.T) {
	c := &Client{}
	add := newAddTool(t, c)
	registry, _ := buildToolRegistry([]ToolDefinition{add})
	ch := &Chat{conv: Conversation(1), tools: registry}

	transport := &fakeTransport{
		streamReplies: [][]StreamChunk{
			// Turn 1: model emits a tool_call mid-stream
			{
				toolCallChunk("add", map[string]any{"A": 2, "B": 3}),
				{Final: true},
			},
			// Turn 2: model streams the post-tool text reply
			{
				textChunk("The sum is 5.", false),
				{Final: true},
			},
		},
	}

	var got strings.Builder
	for chunk, err := range streamIter(ch, transport, `{"role":"user","content":"add 2 and 3"}`) {
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		got.WriteString(chunk.Text)
	}
	if got.String() != "The sum is 5." {
		t.Errorf("post-dispatch text = %q, want %q", got.String(), "The sum is 5.")
	}
	if len(transport.sentMsgs) != 2 {
		t.Fatalf("transport saw %d messages, want 2", len(transport.sentMsgs))
	}
	if !strings.Contains(transport.sentMsgs[1], `"role":"tool"`) {
		t.Errorf("transport[1] missing tool-role result: %s", transport.sentMsgs[1])
	}
}

func TestStream_NonDispatchableToolInforms(t *testing.T) {
	c := &Client{}
	add := newAddTool(t, c)
	registry, _ := buildToolRegistry([]ToolDefinition{
		add,
		NewRawTool("manual_tool", "desc", nil),
	})
	ch := &Chat{conv: Conversation(1), tools: registry}

	transport := &fakeTransport{
		streamReplies: [][]StreamChunk{
			// Turn 1: model calls a tool with no auto-dispatch
			// handler (RawTool) and one the registry doesn't know
			// at all (hallucinated name).
			{
				toolCallChunk("manual_tool", map[string]any{}),
				toolCallChunk("ghost_tool", map[string]any{}),
				{Final: true},
			},
			// Turn 2: model receives the inform-back and replies.
			{
				textChunk("Sorry, neither tool is available.", false),
				{Final: true},
			},
		},
	}

	var got strings.Builder
	for chunk, err := range streamIter(ch, transport, `{}`) {
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		got.WriteString(chunk.Text)
	}
	if !strings.Contains(got.String(), "neither tool is available") {
		t.Errorf("post-inform text = %q, want model recovery line", got.String())
	}
	if len(transport.sentMsgs) != 2 {
		t.Fatalf("transport saw %d messages, want 2", len(transport.sentMsgs))
	}
	informBack := transport.sentMsgs[1]
	if !strings.Contains(informBack, `"role":"tool"`) {
		t.Errorf("transport[1] missing tool-role: %s", informBack)
	}
	if !strings.Contains(informBack, "not auto-dispatchable") {
		t.Errorf("transport[1] missing RawTool inform-back: %s", informBack)
	}
	if !strings.Contains(informBack, "not registered") {
		t.Errorf("transport[1] missing unknown-tool inform-back: %s", informBack)
	}
	// `add` is dispatchable; expect its name surfaced as an alternative.
	if !strings.Contains(informBack, `"available_tools":["add"]`) {
		t.Errorf("transport[1] missing available_tools list: %s", informBack)
	}
}

func TestStream_HopCapExceeded(t *testing.T) {
	c := &Client{}
	add := newAddTool(t, c)
	registry, _ := buildToolRegistry([]ToolDefinition{add})
	ch := &Chat{conv: Conversation(1), tools: registry, maxToolHops: 2}

	loopCall := func() []StreamChunk {
		return []StreamChunk{
			toolCallChunk("add", map[string]any{"A": 1, "B": 1}),
			{Final: true},
		}
	}
	transport := &fakeTransport{
		streamReplies: [][]StreamChunk{
			loopCall(), loopCall(), loopCall(), loopCall(), loopCall(),
		},
	}

	var gotErr error
	for chunk, err := range streamIter(ch, transport, `{}`) {
		if err != nil {
			gotErr = err
			break
		}
		_ = chunk
	}
	if _, ok := errors.AsType[*ToolHopsError](gotErr); !ok {
		t.Errorf("err = %v, want *ToolHopsError", gotErr)
	}
}

func TestStream_ToolCallChunkHasTextToo(t *testing.T) {
	c := &Client{}
	add := newAddTool(t, c)
	registry, _ := buildToolRegistry([]ToolDefinition{add})
	ch := &Chat{conv: Conversation(1), tools: registry}

	// Chunk envelope with BOTH content[text] AND tool_calls in the same envelope.
	mixedChunk := StreamChunk{
		Text: `{"role":"assistant","content":[{"type":"text","text":"Let me add those."}],"tool_calls":[{"type":"function","function":{"name":"add","arguments":{"A":2,"B":3}}}]}`,
	}
	transport := &fakeTransport{
		streamReplies: [][]StreamChunk{
			{mixedChunk, {Final: true}},
			{textChunk("Done: 5.", false), {Final: true}},
		},
	}

	var got strings.Builder
	for chunk, err := range streamIter(ch, transport, `{}`) {
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		got.WriteString(chunk.Text)
	}
	// The pre-dispatch text should appear, then the post-dispatch text.
	if !strings.Contains(got.String(), "Let me add those.") {
		t.Errorf("missing pre-dispatch text: %q", got.String())
	}
	if !strings.Contains(got.String(), "Done: 5.") {
		t.Errorf("missing post-dispatch text: %q", got.String())
	}
	_ = c
}

// streamIter is a tiny adapter that invokes streamWithDispatch as an
// iter.Seq2[Chunk, error]. Tests use this instead of going through
// Chat.SendStream so they can inject a stub transport.
func streamIter(ch *Chat, transport chatTransport, msgJSON string) func(yield func(Chunk, error) bool) {
	return func(yield func(Chunk, error) bool) {
		ch.streamWithDispatch(context.Background(), transport, msgJSON, runtimeConfig{}, yield)
	}
}
