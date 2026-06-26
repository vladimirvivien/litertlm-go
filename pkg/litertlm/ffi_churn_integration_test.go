package litertlm_test

import (
	"context"
	"testing"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

// Regression guard for the FFI return-slot heap-corruption bug.
//
// libffi widens an integral C return to a full register (8 bytes on
// 64-bit) and writes that many bytes through the return pointer, even
// when the C type is narrower. The bindings passed sub-register Go
// slots (int32 counts, uint8 bool predicates) as return slots, so every
// such call overran 4–7 bytes into the adjacent Go allocation. The
// damage accumulated with call count and surfaced as spurious
// "send failed" / "unmapped input" errors or empty replies after a few
// dozen operations. Responses.NumCandidates runs on every generation,
// so any generating workload was exposed.
//
// Each loop below crosses the original failure threshold many times.
// A single error or empty reply means the corruption is back. The fix
// lives in lazyFun.Call (route sub-register returns through a
// register-width buffer), so these guard every narrow-slot binding at
// once.
//
// Gated on LITERTLM_TEST_LIB + LITERTLM_TEST_MODEL; skipped in -short.

func churnClient(t *testing.T, opts ...litertlm.Option) *litertlm.Client {
	t.Helper()
	libDir, modelPath := requireTestModel(t)
	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	base := []litertlm.Option{
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
		litertlm.WithBackend("cpu"),
		litertlm.WithMaxTokens(1024),
	}
	client, err := litertlm.New(context.Background(), append(base, opts...)...)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })
	return client
}

// TestFFIChurn_Generate hammers the session generate path, whose reply
// parsing calls Responses.NumCandidates (a Sint32 return) every time.
func TestFFIChurn_Generate(t *testing.T) {
	client := churnClient(t)
	ctx := context.Background()
	const n = 150
	for i := range n {
		out, err := client.Generate(ctx, "What is the capital of France?",
			litertlm.WithMaxOutputTokens(8))
		if err != nil {
			t.Fatalf("Generate iter %d/%d: %v", i, n, err)
		}
		if out == "" {
			t.Fatalf("Generate iter %d/%d: empty reply (FFI return-slot corruption?)", i, n)
		}
	}
}

// TestFFIChurn_ChatSendClose is the original repro: NewChat + Send +
// Close cycles. SendMessage returns a Sint32 status; before the fix
// this failed partway with "conversation_send_message failed".
func TestFFIChurn_ChatSendClose(t *testing.T) {
	client := churnClient(t)
	ctx := context.Background()
	const n = 150
	for i := range n {
		chat, err := client.NewChat(ctx)
		if err != nil {
			t.Fatalf("NewChat iter %d/%d: %v", i, n, err)
		}
		reply, err := chat.Send(ctx, "What is the capital of France?")
		if err != nil {
			_ = chat.Close()
			t.Fatalf("Send iter %d/%d: %v", i, n, err)
		}
		if reply.Text() == "" {
			_ = chat.Close()
			t.Fatalf("Send iter %d/%d: empty reply (FFI return-slot corruption?)", i, n)
		}
		if _, err := chat.TokenCount(); err != nil {
			_ = chat.Close()
			t.Fatalf("TokenCount iter %d/%d: %v", i, n, err) // Sint32 return
		}
		_ = chat.Close()
	}
}

// TestFFIChurn_ResponseAccessors exercises the Responses accessors with
// the narrowest return slots — NumCandidates (Sint32), Score's
// has-predicate (Uint8), TokenLength (Uint8 + Sint32) — directly and
// repeatedly.
func TestFFIChurn_ResponseAccessors(t *testing.T) {
	client := churnClient(t, litertlm.WithBenchmarkEnabled())
	ctx := context.Background()
	const n = 150
	for i := range n {
		resp, err := client.GenerateResponse(ctx, "Name one primary color.",
			litertlm.WithMaxOutputTokens(4))
		if err != nil {
			t.Fatalf("GenerateResponse iter %d/%d: %v", i, n, err)
		}
		if resp.NumCandidates() < 1 {
			t.Fatalf("GenerateResponse iter %d/%d: %d candidates (corruption?)", i, n, resp.NumCandidates())
		}
		_, _ = resp.Score(0)       // Uint8 has-predicate + Float value
		_, _ = resp.TokenLength(0) // Uint8 has-predicate + Sint32 value
		if b := resp.Benchmark(); b != nil {
			// PrefillTurns / DecodeTurns / per-turn counts are Sint32 returns.
			_ = b.PrefillTurns + b.DecodeTurns
		}
		if resp.Text() == "" {
			t.Fatalf("GenerateResponse iter %d/%d: empty text", i, n)
		}
	}
}

// TestFFIChurn_Tokenize hammers Tokenize (size_t count + borrowed
// pointer) at high count; cheap, so it crosses the threshold by a wide
// margin and also guards the engine token-union getters via TokenLength.
func TestFFIChurn_Tokenize(t *testing.T) {
	client := churnClient(t)
	const n = 1000
	for i := range n {
		toks, err := client.Tokenize("the quick brown fox jumps over the lazy dog")
		if err != nil {
			t.Fatalf("Tokenize iter %d/%d: %v", i, n, err)
		}
		if len(toks) == 0 {
			t.Fatalf("Tokenize iter %d/%d: empty token slice", i, n)
		}
	}
}
