package litertlm

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
)

// fakeTransport scripts replies for the dispatch loop tests. Each
// SendMessage call returns the next entry in replies; sent messages
// are recorded in sentMsgs for assertions. Streaming tests populate
// streamReplies — one []StreamChunk per turn — instead of replies.
type fakeTransport struct {
	mu            sync.Mutex
	replies       []string
	streamReplies [][]StreamChunk
	sentMsgs      []string
	cancelCalls   int
}

func (f *fakeTransport) SendMessage(messageJSON, _ string, _ OptionalArgs) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.sentMsgs = append(f.sentMsgs, messageJSON)
	if len(f.replies) == 0 {
		return "", errors.New("fakeTransport: no scripted reply remaining")
	}
	r := f.replies[0]
	f.replies = f.replies[1:]
	return r, nil
}

func (f *fakeTransport) SendMessageStreamCh(messageJSON, _ string, _ OptionalArgs) <-chan StreamChunk {
	f.mu.Lock()
	f.sentMsgs = append(f.sentMsgs, messageJSON)
	var chunks []StreamChunk
	if len(f.streamReplies) > 0 {
		chunks = f.streamReplies[0]
		f.streamReplies = f.streamReplies[1:]
	}
	f.mu.Unlock()
	out := make(chan StreamChunk, len(chunks)+1)
	for _, c := range chunks {
		out <- c
	}
	close(out)
	return out
}

func (f *fakeTransport) Cancel() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.cancelCalls++
}

func textReply(s string) string {
	return `{"role":"assistant","content":[{"type":"text","text":` + jsonString(s) + `}]}`
}

func toolCallReply(name string, args map[string]any) string {
	argsJSON, _ := json.Marshal(args)
	return `{"role":"assistant","tool_calls":[{"type":"function","function":{"name":` +
		jsonString(name) + `,"arguments":` + string(argsJSON) + `}}]}`
}

func multiToolCallReply(calls ...struct {
	Name string
	Args map[string]any
}) string {
	parts := make([]string, len(calls))
	for i, c := range calls {
		argsJSON, _ := json.Marshal(c.Args)
		parts[i] = `{"type":"function","function":{"name":` +
			jsonString(c.Name) + `,"arguments":` + string(argsJSON) + `}}`
	}
	return `{"role":"assistant","tool_calls":[` + strings.Join(parts, ",") + `]}`
}

func jsonString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

// ---- helpers ------------------------------------------------------------

type addIn struct {
	A int `description:"first addend"`
	B int `description:"second addend"`
}
type addOut struct {
	Sum int `json:"sum"`
}

func newAddTool(t *testing.T, c *Client, opts ...ToolOption) *ManagedTool[addIn, addOut] {
	t.Helper()
	tool, err := RegisterTool(c, "add", "add two ints",
		func(ctx context.Context, in addIn) (addOut, error) {
			return addOut{Sum: in.A + in.B}, nil
		}, opts...)
	if err != nil {
		t.Fatalf("RegisterTool add: %v", err)
	}
	return tool
}

func newFailingTool(t *testing.T, c *Client, name string, fail error, opts ...ToolOption) *ManagedTool[addIn, addOut] {
	t.Helper()
	tool, err := RegisterTool(c, name, "always fails",
		func(ctx context.Context, in addIn) (addOut, error) {
			return addOut{}, fail
		}, opts...)
	if err != nil {
		t.Fatalf("RegisterTool %s: %v", name, err)
	}
	return tool
}

// ---- single-call dispatch -----------------------------------------------

func TestDispatch_SingleCallThenText(t *testing.T) {
	c := &Client{}
	add := newAddTool(t, c)
	registry, _ := buildToolRegistry([]ToolDefinition{add})
	ch := &Chat{tools: registry}
	transport := &fakeTransport{
		replies: []string{
			toolCallReply("add", map[string]any{"a": 17, "b": 25}),
			textReply("The sum is 42."),
		},
	}
	reply, err := ch.send(context.Background(), transport, `{"role":"user","content":"add 17 and 25"}`, OptionalArgs(0))
	if err != nil {
		t.Fatalf("send: %v", err)
	}
	if reply.Text() != "The sum is 42." {
		t.Errorf("Text() = %q", reply.Text())
	}
	if len(transport.sentMsgs) != 2 {
		t.Fatalf("sentMsgs = %d, want 2", len(transport.sentMsgs))
	}
	// Second message should be the tool result.
	if !strings.Contains(transport.sentMsgs[1], `"role":"tool"`) {
		t.Errorf("second message should be tool-role; got %s", transport.sentMsgs[1])
	}
	if !strings.Contains(transport.sentMsgs[1], `"sum":42`) {
		t.Errorf("tool result should contain sum=42; got %s", transport.sentMsgs[1])
	}
}

// ---- multi-call dispatch ------------------------------------------------

func TestDispatch_MultiCallsInOneReply(t *testing.T) {
	c := &Client{}
	add := newAddTool(t, c)
	mul, err := RegisterTool(c, "mul", "multiply",
		func(ctx context.Context, in addIn) (addOut, error) {
			return addOut{Sum: in.A * in.B}, nil
		})
	if err != nil {
		t.Fatalf("RegisterTool mul: %v", err)
	}
	registry, _ := buildToolRegistry([]ToolDefinition{add, mul})
	ch := &Chat{tools: registry}
	transport := &fakeTransport{
		replies: []string{
			multiToolCallReply(
				struct {
					Name string
					Args map[string]any
				}{"add", map[string]any{"a": 1, "b": 2}},
				struct {
					Name string
					Args map[string]any
				}{"mul", map[string]any{"a": 3, "b": 4}},
			),
			textReply("done"),
		},
	}
	if _, err := ch.send(context.Background(), transport, `{"role":"user","content":"go"}`, OptionalArgs(0)); err != nil {
		t.Fatalf("send: %v", err)
	}
	// Second message should bundle both results in one tool-role envelope.
	bundled := transport.sentMsgs[1]
	if !strings.Contains(bundled, `"name":"add"`) || !strings.Contains(bundled, `"name":"mul"`) {
		t.Errorf("bundled tool-role message missing one or both results: %s", bundled)
	}
}

// ---- bail-out cases -----------------------------------------------------

func TestDispatch_BailsOnUnknownTool(t *testing.T) {
	c := &Client{}
	add := newAddTool(t, c)
	registry, _ := buildToolRegistry([]ToolDefinition{add})
	ch := &Chat{tools: registry}
	transport := &fakeTransport{
		replies: []string{toolCallReply("nonexistent", map[string]any{})},
	}
	reply, err := ch.send(context.Background(), transport, `{"role":"user","content":"x"}`, OptionalArgs(0))
	if err != nil {
		t.Fatalf("send: %v", err)
	}
	if !reply.HasToolCalls() {
		t.Error("reply should retain the unknown tool call for manual handling")
	}
	if len(transport.sentMsgs) != 1 {
		t.Errorf("dispatcher should have sent only the original message; sentMsgs = %d", len(transport.sentMsgs))
	}
}

func TestDispatch_BailsOnRawTool(t *testing.T) {
	c := &Client{}
	managed := newAddTool(t, c)
	raw := NewRawTool("manual", "raw", map[string]any{"type": "object"})
	registry, _ := buildToolRegistry([]ToolDefinition{managed, raw})
	ch := &Chat{tools: registry}
	transport := &fakeTransport{
		replies: []string{toolCallReply("manual", map[string]any{})},
	}
	reply, err := ch.send(context.Background(), transport, `{"role":"user","content":"x"}`, OptionalArgs(0))
	if err != nil {
		t.Fatalf("send: %v", err)
	}
	if !reply.HasToolCalls() || reply.ToolCalls()[0].Function.Name != "manual" {
		t.Errorf("expected raw tool call to be returned for manual handling")
	}
}

func TestDispatch_NoManagedToolsRegistered(t *testing.T) {
	ch := &Chat{tools: nil}
	transport := &fakeTransport{
		replies: []string{toolCallReply("anything", map[string]any{})},
	}
	reply, err := ch.send(context.Background(), transport, `{"role":"user","content":"x"}`, OptionalArgs(0))
	if err != nil {
		t.Fatalf("send: %v", err)
	}
	if !reply.HasToolCalls() {
		t.Error("reply should retain tool calls when no managed tools are registered")
	}
}

// ---- ToolPolicy --------------------------------------------------------

func TestDispatch_ToolPolicyReturnOnError(t *testing.T) {
	c := &Client{}
	wantErr := errors.New("boom")
	bad := newFailingTool(t, c, "bad", wantErr) // default policy = ReturnOnError
	registry, _ := buildToolRegistry([]ToolDefinition{bad})
	ch := &Chat{tools: registry}
	transport := &fakeTransport{
		replies: []string{toolCallReply("bad", map[string]any{"a": 1, "b": 2})},
	}
	_, err := ch.send(context.Background(), transport, `{"role":"user","content":"x"}`, OptionalArgs(0))
	if err == nil {
		t.Fatal("expected dispatcher to propagate handler error")
	}
	if !errors.Is(err, wantErr) {
		t.Errorf("err = %v, want wrap of %v", err, wantErr)
	}
}

func TestDispatch_ToolPolicyInformOnError(t *testing.T) {
	c := &Client{}
	wantErr := errors.New("rate limited")
	bad := newFailingTool(t, c, "bad", wantErr, WithToolPolicy(ToolPolicyInformOnError))
	registry, _ := buildToolRegistry([]ToolDefinition{bad})
	ch := &Chat{tools: registry}
	transport := &fakeTransport{
		replies: []string{
			toolCallReply("bad", map[string]any{"a": 1, "b": 2}),
			textReply("Sorry, please retry later."),
		},
	}
	reply, err := ch.send(context.Background(), transport, `{"role":"user","content":"x"}`, OptionalArgs(0))
	if err != nil {
		t.Fatalf("send: %v", err)
	}
	if reply.Text() != "Sorry, please retry later." {
		t.Errorf("expected loop to continue after informing model; got %q", reply.Text())
	}
	if len(transport.sentMsgs) != 2 {
		t.Fatalf("sentMsgs = %d, want 2", len(transport.sentMsgs))
	}
	if !strings.Contains(transport.sentMsgs[1], `"error":"rate limited"`) {
		t.Errorf("tool-role message should carry the error payload; got %s", transport.sentMsgs[1])
	}
}

// ---- WithMaxToolHops ----------------------------------------------------

func TestDispatch_HopsExceeded(t *testing.T) {
	c := &Client{}
	add := newAddTool(t, c)
	registry, _ := buildToolRegistry([]ToolDefinition{add})
	ch := &Chat{tools: registry, maxToolHops: 2}
	// Model never stops calling the tool; loop should terminate after
	// 2 dispatch hops (3 model turns).
	transport := &fakeTransport{
		replies: []string{
			toolCallReply("add", map[string]any{"a": 1, "b": 1}),
			toolCallReply("add", map[string]any{"a": 2, "b": 2}),
			toolCallReply("add", map[string]any{"a": 3, "b": 3}),
		},
	}
	_, err := ch.send(context.Background(), transport, `{"role":"user","content":"x"}`, OptionalArgs(0))
	if err == nil {
		t.Fatal("expected ErrToolHopsExceeded")
	}
	if !errors.Is(err, ErrToolHopsExceeded) {
		t.Errorf("err = %v, want errors.Is ErrToolHopsExceeded", err)
	}
	var hopsErr *ToolHopsError
	if !errors.As(err, &hopsErr) {
		t.Fatal("err should match *ToolHopsError")
	}
	if hopsErr.Hops != 2 {
		t.Errorf("Hops = %d, want 2", hopsErr.Hops)
	}
	if hopsErr.LastReply == nil || !hopsErr.LastReply.HasToolCalls() {
		t.Error("LastReply should carry the final tool-call reply")
	}
}

// ---- option ------------------------------------------------------------

func TestWithMaxToolHops_IgnoresNonPositive(t *testing.T) {
	cfg := chatConfig{}
	WithMaxToolHops(0)(&cfg)
	if cfg.maxToolHops != 0 {
		t.Errorf("0 should be ignored; got %d", cfg.maxToolHops)
	}
	WithMaxToolHops(-3)(&cfg)
	if cfg.maxToolHops != 0 {
		t.Errorf("negative should be ignored; got %d", cfg.maxToolHops)
	}
	WithMaxToolHops(7)(&cfg)
	if cfg.maxToolHops != 7 {
		t.Errorf("positive should set; got %d", cfg.maxToolHops)
	}
}
