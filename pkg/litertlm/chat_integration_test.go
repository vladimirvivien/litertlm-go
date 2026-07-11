package litertlm_test

import (
	"context"
	"strings"
	"testing"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func TestChat_Integration_Send(t *testing.T) {
	libDir, modelPath := requireTestModel(t)

	if err := litertlm.Load(libDir, "cpu", ""); err != nil {
		t.Fatalf("Load: %v", err)
	}

	client, err := litertlm.New(context.Background(),
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
	)
	if err != nil {
		t.Fatalf("New client: %v", err)
	}
	defer func() { _ = client.Close() }()

	ctx := context.Background()
	chat, err := client.NewChat(ctx, litertlm.WithSystemPrompt("You are a helpful assistant."))
	if err != nil {
		t.Fatalf("NewChat: %v", err)
	}
	defer func() { _ = chat.Close() }()

	// 1. Test Send
	reply, err := chat.Send(ctx, "Reply with the single word: Paris")
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	if reply.Text() == "" {
		t.Fatal("empty reply from Send")
	}
	if !strings.Contains(strings.ToLower(reply.Text()), "paris") {
		t.Errorf("expected reply to contain 'paris', got: %q", reply.Text())
	}

	// 2. Test template introspection on Chat
	preface, err := chat.Conversation().RenderPreface()
	if err != nil {
		t.Fatalf("Conversation.RenderPreface: %v", err)
	}
	if preface == "" {
		t.Fatal("RenderPreface returned empty string")
	}
	if !strings.Contains(preface, "helpful assistant") {
		t.Errorf("expected rendered preface to contain system prompt, got: %q", preface)
	}
}

func TestChat_Integration_OptionsAndStream(t *testing.T) {
	libDir, modelPath := requireTestModel(t)

	if err := litertlm.Load(libDir, "cpu", ""); err != nil {
		t.Fatalf("Load: %v", err)
	}

	client, err := litertlm.New(context.Background(),
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
	)
	if err != nil {
		t.Fatalf("New client: %v", err)
	}
	defer func() { _ = client.Close() }()

	ctx := context.Background()

	// 1. Test WithMaxOutputTokens limiting output tokens on Send
	chat, err := client.NewChat(ctx)
	if err != nil {
		t.Fatalf("NewChat: %v", err)
	}
	defer func() { _ = chat.Close() }()

	reply, err := chat.Send(ctx, "Explain photosynthesis in detail.", litertlm.WithMaxOutputTokens(4))
	if err != nil {
		t.Fatalf("Send with max tokens: %v", err)
	}
	// We expect the reply to be very short (under 4-10 tokens depending on tokenizer boundaries)
	if len(strings.Split(reply.Text(), " ")) > 15 {
		t.Errorf("expected output to be constrained, got long text: %q", reply.Text())
	}

	// 2. Test SendStream
	streamChat, err := client.NewChat(ctx)
	if err != nil {
		t.Fatalf("NewChat for stream: %v", err)
	}
	defer func() { _ = streamChat.Close() }()

	var chunks []string
	for chunk, err := range streamChat.SendStream(ctx, "Name three primary colors in one sentence.") {
		if err != nil {
			t.Fatalf("stream error: %v", err)
		}
		chunks = append(chunks, chunk.Text)
	}
	streamOutput := strings.Join(chunks, "")
	if streamOutput == "" {
		t.Fatal("stream yielded no output")
	}
}
