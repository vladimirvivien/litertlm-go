package litertgo_test

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vladimirvivien/litert-go/lm"
	"github.com/vladimirvivien/litertlm-go/backend/litertgo"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

// Integration tests drive the full litertlm-go Client / Chat API over
// the litert-go engine. They need the LiteRT runtime libraries and a
// .litertlm model:
//
//	LITERT_LIB          = LiteRT runtime shared-library directory
//	LITERTLM_TEST_MODEL = absolute path to a .litertlm file
//
// Both unset (or -short) skips.

func openBackend(t *testing.T) (*litertgo.Backend, string) {
	t.Helper()
	if testing.Short() {
		t.Skip("short mode")
	}
	libDir := os.Getenv("LITERT_LIB")
	modelPath := os.Getenv("LITERTLM_TEST_MODEL")
	if libDir == "" || modelPath == "" {
		t.Skip("LITERT_LIB / LITERTLM_TEST_MODEL not set")
	}
	b, err := litertgo.Open(context.Background(), modelPath, lm.WithLibDir(libDir))
	if err != nil {
		t.Fatalf("litertgo.Open: %v", err)
	}
	return b, modelPath
}

func newClient(t *testing.T) (*litertlm.Client, string) {
	t.Helper()
	b, modelPath := openBackend(t)
	c, err := litertlm.New(context.Background(), litertlm.WithEngineBackend(b))
	if err != nil {
		b.Close()
		t.Fatalf("litertlm.New(WithEngineBackend): %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c, modelPath
}

func TestGoBackend_ChatSend(t *testing.T) {
	c, modelPath := newClient(t)

	ch, err := c.NewChat(context.Background())
	if err != nil {
		t.Fatalf("NewChat: %v", err)
	}
	defer ch.Close()

	reply, err := ch.Send(context.Background(), "Name one primary color.")
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	text := reply.Text()
	if text == "" {
		t.Fatal("Send returned empty reply")
	}
	t.Logf("reply: %q", text)

	// gemma3-270m greedy is the cross-engine byte-exact anchor (the
	// engdiff baseline); pin its reply when that model is under test.
	if strings.Contains(filepath.Base(modelPath), "gemma3-270m") {
		const want = "The primary color is **blue**.\n"
		if text != want {
			t.Errorf("gemma3-270m greedy reply = %q, want %q (cross-engine anchor)", text, want)
		}
	}
}

func TestGoBackend_ChatSendStream_MatchesSend(t *testing.T) {
	c, _ := newClient(t)

	// Fresh chats with the same greedy prompt must agree between the
	// streaming and non-streaming paths.
	ch1, err := c.NewChat(context.Background())
	if err != nil {
		t.Fatalf("NewChat: %v", err)
	}
	defer ch1.Close()
	reply, err := ch1.Send(context.Background(), "Name one primary color.")
	if err != nil {
		t.Fatalf("Send: %v", err)
	}

	ch2, err := c.NewChat(context.Background())
	if err != nil {
		t.Fatalf("NewChat: %v", err)
	}
	defer ch2.Close()
	var streamed strings.Builder
	for chunk, err := range ch2.SendStream(context.Background(), "Name one primary color.") {
		if err != nil {
			t.Fatalf("SendStream: %v", err)
		}
		streamed.WriteString(chunk.Text)
	}

	if streamed.String() != reply.Text() {
		t.Errorf("stream concat = %q, Send = %q", streamed.String(), reply.Text())
	}
}

func TestGoBackend_Generate_RawPrompt(t *testing.T) {
	c, _ := newClient(t)

	out, err := c.Generate(context.Background(), "The capital of France is",
		litertlm.WithMaxOutputTokens(16))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if out == "" {
		t.Fatal("Generate returned empty output")
	}
	t.Logf("generate: %q", out)
}

func TestGoBackend_Tokenize(t *testing.T) {
	c, _ := newClient(t)

	toks, err := c.Tokenize("hello world")
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}
	if len(toks) == 0 {
		t.Fatal("Tokenize returned no tokens")
	}
	n, err := c.TokenLength("hello world")
	if err != nil || n != len(toks) {
		t.Errorf("TokenLength = %d, %v; want %d, nil", n, err, len(toks))
	}
}

func TestGoBackend_ContextCancel(t *testing.T) {
	c, _ := newClient(t)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := c.Generate(ctx, "write a very long story"); err == nil {
		t.Error("Generate with cancelled ctx: want error, got nil")
	}
}
