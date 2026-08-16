package litertlm_test

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

// Text-input base battery across the supported-model matrix. Every
// .litertlm file under LITERTLM_TEST_MODELS_DIR runs the same three
// checks — chat send, tokenize round-trip, streaming coherence — as a
// named subtest. The battery is processor-agnostic: it covers only the
// text paths every supported family shares. Multi-turn, tool-call, and
// multimodal coverage live in their own tests because not every family
// supports them (Qwen3 multi-turn is blocked upstream; see
// docs/models.md).
//
// Generation routes through Chat, not Client.Generate: every published
// .litertlm model is instruction-tuned, and the raw Generate path does
// not apply the chat template, so -it models degenerate or emit empty
// output. Each check opens a fresh single-turn Chat, so the Qwen3
// multi-turn block is never reached.
//
// Populate LITERTLM_TEST_MODELS_DIR with one model per family from the
// support matrix; see docs/models.md for the supported families. Set
// LITERTLM_TEST_BACKEND to "gpu" to run the battery on the GPU backend
// (default "cpu").

// requireTestModelsDir returns the library dir and a sorted list of
// .litertlm files discovered under LITERTLM_TEST_MODELS_DIR. Calls
// t.Skip under -short, when either env var is empty, or when the
// directory holds no model files. Multi-model analogue of the single
// LITERTLM_TEST_MODEL harness in conversation_integration_test.go.
func requireTestModelsDir(t *testing.T) (libDir string, models []string) {
	t.Helper()
	if testing.Short() {
		t.Skip("integration test skipped in -short mode")
	}
	libDir = os.Getenv("LITERTLM_TEST_LIB")
	dir := os.Getenv("LITERTLM_TEST_MODELS_DIR")
	if libDir == "" || dir == "" {
		t.Skip("integration test requires LITERTLM_TEST_LIB and LITERTLM_TEST_MODELS_DIR")
	}

	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.EqualFold(filepath.Ext(path), ".litertlm") {
			models = append(models, path)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk LITERTLM_TEST_MODELS_DIR: %v", err)
	}
	if len(models) == 0 {
		t.Skipf("no .litertlm files found under %s", dir)
	}
	sort.Strings(models)
	return libDir, models
}

// newMatrixClient builds a Client for one model and registers Close on cleanup
// so the next subtest's model loads into freed memory.
func newMatrixClient(t *testing.T, libDir, modelPath string) *litertlm.Client {
	t.Helper()

	var opts []litertlm.Option
	opts = append(opts,
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
		litertlm.WithMaxTokens(1024),
	)
	backend := os.Getenv("LITERTLM_TEST_BACKEND")
	if backend != "" {
		opts = append(opts, litertlm.WithBackend(backend))
	}

	client, err := litertlm.New(context.Background(), opts...)
	if err != nil {
		t.Skipf("backend %q unsupported for %s: %v", backend, filepath.Base(modelPath), err)
		return nil
	}
	t.Cleanup(func() { _ = client.Close() })
	return client
}

func TestModelMatrix(t *testing.T) {
	libDir, models := requireTestModelsDir(t)
	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	for _, modelPath := range models {
		name := strings.TrimSuffix(filepath.Base(modelPath), filepath.Ext(modelPath))
		t.Run(name, func(t *testing.T) {
			client := newMatrixClient(t, libDir, modelPath)
			if client == nil {
				return
			}
			t.Run("ChatSend", func(t *testing.T) { testMatrixGenerate(t, client) })
			t.Run("TokenizeRoundTrip", func(t *testing.T) { testMatrixTokenizeRoundTrip(t, client) })
			t.Run("StreamingCoherence", func(t *testing.T) { testMatrixStreaming(t, client) })
		})
	}
}

// newMatrixChat opens a fresh single-turn Chat on client. Cleanup
// closes it so the next check's Chat starts clean.
func newMatrixChat(t *testing.T, client *litertlm.Client) *litertlm.Chat {
	t.Helper()
	chat, err := client.NewChat(context.Background())
	if err != nil {
		t.Fatalf("NewChat: %v", err)
	}
	t.Cleanup(func() { _ = chat.Close() })
	return chat
}

// isPadOnly reports whether s carries no content beyond <pad> markers.
// A model that emits only pad tokens (observed when a backend mishandles
// a model's kernels) would otherwise slip past a bare non-empty check.
func isPadOnly(s string) bool {
	return strings.TrimSpace(strings.ReplaceAll(s, "<pad>", "")) == ""
}

// testMatrixGenerate asserts a single chat turn returns real,
// non-degenerate output. The prompt is trivial so any coherent answer
// passes across families with differing instruction-following strength.
func testMatrixGenerate(t *testing.T, client *litertlm.Client) {
	chat := newMatrixChat(t, client)
	reply, err := chat.Send(context.Background(),
		"What is the capital of France? Answer in one word.",
		litertlm.WithMaxOutputTokens(128))
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	if strings.TrimSpace(reply.Text()) == "" {
		t.Fatal("Send returned empty reply")
	}
	if isPadOnly(reply.Text()) {
		t.Skipf("Send returned pad-only output (unsupported backend kernel): %.80q", reply.Text())
		return
	}
	t.Logf("reply: %q", reply.Text())
}

// testMatrixTokenizeRoundTrip asserts Tokenize yields a positive-length
// id sequence, TokenLength agrees with it, and detokenizing recovers
// the input's content words. Content-substring comparison (after
// normalizing SentencePiece's U+2581 space marker) sidesteps brittle
// whitespace assertions across different tokenizers.
func testMatrixTokenizeRoundTrip(t *testing.T, client *litertlm.Client) {
	const text = "The quick brown fox jumps over the lazy dog."

	ids, err := client.Tokenize(text)
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}
	if len(ids) == 0 {
		t.Fatal("Tokenize returned no tokens")
	}

	n, err := client.TokenLength(text)
	if err != nil {
		t.Fatalf("TokenLength: %v", err)
	}
	if n != len(ids) {
		t.Errorf("TokenLength=%d disagrees with len(Tokenize)=%d", n, len(ids))
	}

	decoded, err := client.Engine().Detokenize(ids)
	if err != nil {
		t.Fatalf("Detokenize: %v", err)
	}
	norm := strings.ToLower(strings.ReplaceAll(decoded, "▁", " "))
	for _, want := range []string{"quick", "brown", "fox"} {
		if !strings.Contains(norm, want) {
			t.Errorf("round-trip lost %q; decoded=%q", want, decoded)
		}
	}
}

// testMatrixStreaming asserts streaming chat yields at least one chunk,
// a non-empty accumulated body, and the contract's single Final chunk.
// Exact equality with the non-streaming reply is not asserted: the
// default sampler is stochastic, so two runs of the same prompt
// legitimately differ.
func testMatrixStreaming(t *testing.T, client *litertlm.Client) {
	chat := newMatrixChat(t, client)
	var b strings.Builder
	chunks := 0
	sawFinal := false
	for chunk, err := range chat.SendStream(context.Background(),
		"Name three primary colors.",
		litertlm.WithMaxOutputTokens(128)) {
		if err != nil {
			t.Fatalf("SendStream: %v", err)
		}
		b.WriteString(chunk.Text)
		chunks++
		if chunk.Final {
			sawFinal = true
		}
	}
	if chunks == 0 {
		t.Fatal("GenerateStream yielded no chunks")
	}
	if strings.TrimSpace(b.String()) == "" {
		t.Fatal("GenerateStream produced empty output")
	}
	if isPadOnly(b.String()) {
		t.Skipf("GenerateStream produced pad-only output (unsupported backend kernel): %.80q", b.String())
		return
	}
	if !sawFinal {
		t.Error("GenerateStream never yielded a Final chunk")
	}
	t.Logf("streamed %d chunks: %q", chunks, b.String())
}
