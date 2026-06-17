// Package backendtest is the seam conformance suite: a fixed set of
// behavioral checks every litertlm.Backend implementation must pass,
// run against a live model. Backend packages call Run from an
// integration test, supplying only a constructor; the checks
// themselves are engine-agnostic and assert mechanics (candidates
// arrive, stream concatenation equals the synchronous result under
// greedy decoding, envelopes parse, cancellation terminates streams)
// rather than model-specific text.
package backendtest

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

// prompt is answerable by every model in the validation zoo and short
// enough that greedy replies end at EOS quickly.
const prompt = "Name one primary color."

// Run executes the conformance checks against a Backend. open is
// called once per subtest so each check gets a fresh engine-backed
// instance; Run closes what it opens.
func Run(t *testing.T, open func(tb testing.TB) litertlm.Backend) {
	t.Run("Tokenize", func(t *testing.T) {
		b := open(t)
		defer b.Close()
		toks, err := b.Tokenize("hello world")
		if err != nil {
			t.Fatalf("Tokenize: %v", err)
		}
		if len(toks) == 0 {
			t.Fatal("Tokenize returned no tokens")
		}
	})

	t.Run("SessionGenerate", func(t *testing.T) {
		b := open(t)
		defer b.Close()
		sess := newSession(t, b)
		defer sess.Close()

		cands, err := sess.Generate([]litertlm.Part{litertlm.Text(prompt)})
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if len(cands) == 0 || cands[0].Text == "" {
			t.Fatalf("Generate candidates = %+v, want at least one with text", cands)
		}
	})

	t.Run("SessionStreamMatchesGenerate", func(t *testing.T) {
		b := open(t)
		defer b.Close()

		sync := newSession(t, b)
		cands, err := sync.Generate([]litertlm.Part{litertlm.Text(prompt)})
		sync.Close()
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if len(cands) == 0 {
			t.Fatal("Generate returned no candidates")
		}

		stream := newSession(t, b)
		defer stream.Close()
		var got strings.Builder
		for sc := range stream.GenerateStreamCh([]litertlm.Part{litertlm.Text(prompt)}) {
			if sc.Err != nil {
				t.Fatalf("stream chunk error: %v", sc.Err)
			}
			got.WriteString(sc.Text)
		}
		if got.String() != cands[0].Text {
			t.Errorf("greedy stream concat = %q, Generate = %q", got.String(), cands[0].Text)
		}
	})

	t.Run("ChatSendMessage", func(t *testing.T) {
		b := open(t)
		defer b.Close()
		tr := newTransport(t, b)
		defer tr.Close()

		raw, err := tr.SendMessage(userEnvelope(t, prompt), "", litertlm.RuntimeArgs{})
		if err != nil {
			t.Fatalf("SendMessage: %v", err)
		}
		if text := assistantText(t, raw); text == "" {
			t.Fatalf("assistant envelope has no text: %s", raw)
		}
	})

	t.Run("ChatStreamMatchesSend", func(t *testing.T) {
		b := open(t)
		defer b.Close()

		sync := newTransport(t, b)
		raw, err := sync.SendMessage(userEnvelope(t, prompt), "", litertlm.RuntimeArgs{})
		sync.Close()
		if err != nil {
			t.Fatalf("SendMessage: %v", err)
		}
		want := assistantText(t, raw)

		stream := newTransport(t, b)
		defer stream.Close()
		var got strings.Builder
		for sc := range stream.SendMessageStreamCh(userEnvelope(t, prompt), "", litertlm.RuntimeArgs{}) {
			if sc.Err != nil {
				t.Fatalf("stream chunk error: %v", sc.Err)
			}
			got.WriteString(streamText(sc.Text))
		}
		if got.String() != want {
			t.Errorf("greedy chat stream concat = %q, SendMessage = %q", got.String(), want)
		}
	})

	t.Run("ChatMultiTurn", func(t *testing.T) {
		b := open(t)
		defer b.Close()
		tr := newTransport(t, b)
		defer tr.Close()

		if _, err := tr.SendMessage(userEnvelope(t, "My name is Alice."), "", litertlm.RuntimeArgs{}); err != nil {
			t.Fatalf("turn 1: %v", err)
		}
		raw, err := tr.SendMessage(userEnvelope(t, "What is my name?"), "", litertlm.RuntimeArgs{})
		if err != nil {
			t.Fatalf("turn 2: %v", err)
		}
		if text := assistantText(t, raw); text == "" {
			t.Fatalf("turn 2 envelope has no text: %s", raw)
		}
	})

	t.Run("ChatWithHistory", func(t *testing.T) {
		b := open(t)
		defer b.Close()

		hist := []map[string]string{
			{"role": "user", "content": "My name is Bob. Remember it."},
			{"role": "assistant", "content": "Hello Bob! I will remember your name."},
		}
		histJSON, err := json.Marshal(hist)
		if err != nil {
			t.Fatalf("marshal history: %v", err)
		}

		tr, err := b.NewChatTransport(litertlm.ConversationSetup{
			MessagesJSON:    string(histJSON),
			MaxOutputTokens: 128,
		})
		if err != nil {
			t.Fatalf("NewChatTransport with history: %v", err)
		}
		defer tr.Close()

		tc, err := tr.TokenCount()
		if err != nil {
			t.Fatalf("TokenCount: %v", err)
		}
		if tc <= 0 {
			t.Errorf("TokenCount = %d, want > 0 after history initialization", tc)
		}

		raw, err := tr.SendMessage(userEnvelope(t, "What is my name?"), "", litertlm.RuntimeArgs{})
		if err != nil {
			t.Fatalf("SendMessage: %v", err)
		}
		text := assistantText(t, raw)
		if text == "" {
			t.Fatalf("reply envelope has no text: %s", raw)
		}
		if !strings.Contains(text, "Bob") {
			t.Logf("note: reply after history did not echo name (model-dependent): %q", text)
		}
	})

	t.Run("SessionCancelTerminatesStream", func(t *testing.T) {
		b := open(t)
		defer b.Close()
		sess := newSession(t, b)
		defer sess.Close()

		ch := sess.GenerateStreamCh([]litertlm.Part{litertlm.Text("Write a very long story about the sea.")})
		// Consume one chunk so generation is demonstrably in flight,
		// then cancel; the channel must close (drain tolerates a
		// trailing error chunk).
		<-ch
		sess.Cancel()
		for range ch {
		}
	})
}

func newSession(t *testing.T, b litertlm.Backend) litertlm.SessionBackend {
	t.Helper()
	sess, err := b.NewSessionBackend(litertlm.SessionSetup{MaxOutputTokens: 128})
	if err != nil {
		t.Fatalf("NewSessionBackend: %v", err)
	}
	return sess
}

func newTransport(t *testing.T, b litertlm.Backend) litertlm.ChatTransport {
	t.Helper()
	tr, err := b.NewChatTransport(litertlm.ConversationSetup{})
	if err != nil {
		t.Fatalf("NewChatTransport: %v", err)
	}
	return tr
}

func userEnvelope(t *testing.T, text string) string {
	t.Helper()
	b, err := json.Marshal(map[string]string{"role": "user", "content": text})
	if err != nil {
		t.Fatalf("marshal user envelope: %v", err)
	}
	return string(b)
}

// assistantText extracts the concatenated text parts from an
// assistant reply envelope.
func assistantText(t *testing.T, raw string) string {
	t.Helper()
	var msg struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("parse assistant envelope: %v (raw=%s)", err, raw)
	}
	var b strings.Builder
	for _, p := range msg.Content {
		if p.Type == "text" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// streamText mirrors the Chat dispatch loop's chunk handling: a JSON
// envelope chunk contributes its text parts, anything else passes
// through raw.
func streamText(raw string) string {
	if raw == "" {
		return ""
	}
	var msg struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		return raw
	}
	var b strings.Builder
	for _, p := range msg.Content {
		if p.Type == "text" {
			b.WriteString(p.Text)
		}
	}
	return b.String()
}
