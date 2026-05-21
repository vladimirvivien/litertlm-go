package litertlm

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

// SendMulti and SendMultiStream funnel through partsToConversationMessage
// and Chat.send / Conversation.SendMessageStreamCh. The tests below
// verify the JSON shape produced for multimodal inputs, the
// degenerate empty-parts case, and the closed-chat guard. Dispatch-
// loop coverage is shared with the text path; see chat_dispatch_test.go.

func TestSendMulti_TextOnlyMatchesSend(t *testing.T) {
	msg, err := partsToConversationMessage([]Part{Text("hello world")})
	if err != nil {
		t.Fatalf("partsToConversationMessage: %v", err)
	}
	var got struct {
		Role    string `json:"role"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal([]byte(msg), &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.Role != "user" {
		t.Errorf("role = %q, want %q", got.Role, "user")
	}
	if len(got.Content) != 1 || got.Content[0].Type != "text" || got.Content[0].Text != "hello world" {
		t.Errorf("content = %+v, want [{text hello world}]", got.Content)
	}
}

func TestSendMulti_ImagePlusTextEnvelope(t *testing.T) {
	img := Image([]byte{0x89, 'P', 'N', 'G'})
	msg, err := partsToConversationMessage([]Part{img, Text("describe")})
	if err != nil {
		t.Fatalf("partsToConversationMessage: %v", err)
	}
	var got struct {
		Role    string `json:"role"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
			Blob string `json:"blob"`
		} `json:"content"`
	}
	if err := json.Unmarshal([]byte(msg), &got); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got.Role != "user" {
		t.Errorf("role = %q, want %q", got.Role, "user")
	}
	if len(got.Content) != 2 {
		t.Fatalf("len(content) = %d, want 2", len(got.Content))
	}
	if got.Content[0].Type != "image" || got.Content[0].Blob == "" {
		t.Errorf("content[0] = %+v, want image with blob", got.Content[0])
	}
	if got.Content[1].Type != "text" || got.Content[1].Text != "describe" {
		t.Errorf("content[1] = %+v, want text=describe", got.Content[1])
	}
}

func TestSendMulti_EmptyParts(t *testing.T) {
	ch := &Chat{conv: Conversation(1)}
	if _, err := ch.SendMulti(context.Background(), nil); err == nil ||
		!strings.Contains(err.Error(), "empty parts") {
		t.Errorf("SendMulti(nil) err = %v, want 'empty parts'", err)
	}
	if _, err := ch.SendMulti(context.Background(), []Part{}); err == nil ||
		!strings.Contains(err.Error(), "empty parts") {
		t.Errorf("SendMulti([]) err = %v, want 'empty parts'", err)
	}
}

func TestSendMultiStream_EmptyParts(t *testing.T) {
	ch := &Chat{conv: Conversation(1)}
	var gotErr error
	for _, err := range ch.SendMultiStream(context.Background(), nil) {
		gotErr = err
		break
	}
	if gotErr == nil || !strings.Contains(gotErr.Error(), "empty parts") {
		t.Errorf("SendMultiStream(nil) err = %v, want 'empty parts'", gotErr)
	}
}

func TestSendMulti_ClosedChat(t *testing.T) {
	ch := &Chat{closed: true}
	_, err := ch.SendMulti(context.Background(), []Part{Text("hi")})
	if err == nil || !strings.Contains(err.Error(), "Chat is closed") {
		t.Errorf("SendMulti err = %v, want 'Chat is closed'", err)
	}
}

func TestSendMultiStream_ClosedChat(t *testing.T) {
	ch := &Chat{closed: true}
	var gotErr error
	for _, err := range ch.SendMultiStream(context.Background(), []Part{Text("hi")}) {
		gotErr = err
		break
	}
	if gotErr == nil || !strings.Contains(gotErr.Error(), "Chat is closed") {
		t.Errorf("SendMultiStream err = %v, want 'Chat is closed'", gotErr)
	}
}

// TestSendMulti_DispatchLoopMultimodal verifies that the dispatch
// loop is exercised on multimodal input — the transport receives the
// multimodal envelope, and the loop returns the assistant reply.
func TestSendMulti_DispatchLoopMultimodal(t *testing.T) {
	transport := &fakeTransport{
		replies: []string{textReply("a wooden table with a lamp")},
	}
	ch := &Chat{conv: Conversation(1)} // non-zero so checkOpen passes
	ctx := context.Background()
	msgJSON, err := partsToConversationMessage([]Part{
		Image([]byte{0x89, 'P', 'N', 'G'}),
		Text("describe this image"),
	})
	if err != nil {
		t.Fatalf("partsToConversationMessage: %v", err)
	}
	reply, err := ch.send(ctx, transport, msgJSON, OptionalArgs(0), false)
	if err != nil {
		t.Fatalf("ch.send: %v", err)
	}
	if reply.Text() != "a wooden table with a lamp" {
		t.Errorf("reply = %q, want 'a wooden table with a lamp'", reply.Text())
	}
	if len(transport.sentMsgs) != 1 {
		t.Fatalf("transport saw %d messages, want 1", len(transport.sentMsgs))
	}
	if !strings.Contains(transport.sentMsgs[0], `"type":"image"`) {
		t.Errorf("transport received non-multimodal message: %s", transport.sentMsgs[0])
	}
	if !strings.Contains(transport.sentMsgs[0], `"text":"describe this image"`) {
		t.Errorf("transport message missing text part: %s", transport.sentMsgs[0])
	}
}

// TestSendMulti_TransportError verifies error propagation through
// the multimodal path.
func TestSendMulti_TransportError(t *testing.T) {
	transport := &fakeTransport{} // empty replies → SendMessage returns error
	ch := &Chat{conv: Conversation(1)}
	msgJSON, _ := partsToConversationMessage([]Part{Text("hi")})
	if _, err := ch.send(context.Background(), transport, msgJSON, OptionalArgs(0), false); err == nil {
		t.Fatal("expected error, got nil")
	}
}
