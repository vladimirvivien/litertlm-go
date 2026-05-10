package litertlm

import (
	"strings"
	"testing"
)

// Client.Tokenize / TokenLength are thin pass-throughs to Engine.Tokenize,
// which requires a loaded model — covered end-to-end in examples/bot.
// The unit-test surface here is the routing: a Client with a zero
// Engine handle must surface the engine's "invalid engine" error
// rather than panic.
func TestClientTokenize_ZeroEngine(t *testing.T) {
	c := &Client{}

	if _, err := c.Tokenize("hello"); err == nil {
		t.Fatalf("Tokenize on zero-engine Client: want error, got nil")
	} else if !strings.Contains(err.Error(), "invalid engine") {
		t.Fatalf("Tokenize error = %v, want wrapping invalid engine", err)
	}

	if _, err := c.TokenLength("hello"); err == nil {
		t.Fatalf("TokenLength on zero-engine Client: want error, got nil")
	} else if !strings.Contains(err.Error(), "invalid engine") {
		t.Fatalf("TokenLength error = %v, want wrapping invalid engine", err)
	}
}
