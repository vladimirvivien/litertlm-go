package litertgo_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/vladimirvivien/litert-go/lm"
	"github.com/vladimirvivien/litertlm-go/backend/litertgo"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

// Multimodal tests need a gemma-4 family model with vision/audio
// towers: LITERT_LIB + LITERTLM_TEST_GEMMA4_MODEL. Skips otherwise.
func newGemma4Client(t testing.TB) *litertlm.Client {
	t.Helper()
	if testing.Short() {
		t.Skip("short mode")
	}
	libDir := os.Getenv("LITERT_LIB")
	modelPath := os.Getenv("LITERTLM_TEST_GEMMA4_MODEL")
	if libDir == "" || modelPath == "" {
		t.Skip("LITERT_LIB / LITERTLM_TEST_GEMMA4_MODEL not set")
	}
	b, err := litertgo.Open(context.Background(), modelPath, lm.WithLibDir(libDir))
	if err != nil {
		t.Fatalf("litertgo.Open: %v", err)
	}
	c, err := litertlm.New(context.Background(), litertlm.WithEngineBackend(b))
	if err != nil {
		b.Close()
		t.Fatalf("litertlm.New: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func testImage(t *testing.T) []byte {
	t.Helper()
	data, err := os.ReadFile("../../examples/testdata/img1.png")
	if err != nil {
		t.Skipf("test image unavailable: %v", err)
	}
	return data
}

func TestGoBackend_GenerateMulti_Image(t *testing.T) {
	c := newGemma4Client(t)

	out, err := c.GenerateMulti(context.Background(), []litertlm.Part{
		litertlm.Image(testImage(t)),
		litertlm.Text("Describe this image in one sentence."),
	}, litertlm.WithMaxOutputTokens(64))
	if err != nil {
		t.Fatalf("GenerateMulti: %v", err)
	}
	if out == "" {
		t.Fatal("empty image description")
	}
	t.Logf("image: %q", out)
}

func TestGoBackend_GenerateMultiStream_Image(t *testing.T) {
	c := newGemma4Client(t)

	var got strings.Builder
	for chunk, err := range c.GenerateMultiStream(context.Background(), []litertlm.Part{
		litertlm.Image(testImage(t)),
		litertlm.Text("Describe this image in one sentence."),
	}, litertlm.WithMaxOutputTokens(64)) {
		if err != nil {
			t.Fatalf("GenerateMultiStream: %v", err)
		}
		got.WriteString(chunk.Text)
	}
	if got.String() == "" {
		t.Fatal("empty streamed image description")
	}
	t.Logf("image stream: %q", got.String())
}

func TestGoBackend_GenerateMulti_Audio(t *testing.T) {
	c := newGemma4Client(t)

	out, err := c.GenerateMulti(context.Background(), []litertlm.Part{
		litertlm.Audio(toneWAV(t, 2, 440)),
		litertlm.Text("Describe what you hear in one sentence."),
	}, litertlm.WithMaxOutputTokens(64))
	if err != nil {
		t.Fatalf("GenerateMulti: %v", err)
	}
	if out == "" {
		t.Fatal("empty audio description")
	}
	t.Logf("audio: %q", out)
}

func TestGoBackend_MultimodalMidConversation(t *testing.T) {
	c := newGemma4Client(t)

	chat, err := c.NewChat(context.Background())
	if err != nil {
		t.Fatalf("NewChat: %v", err)
	}
	defer chat.Close()
	if _, err := chat.Send(context.Background(), "Hello."); err != nil {
		t.Fatalf("Send: %v", err)
	}
	_, err = chat.SendMulti(context.Background(), []litertlm.Part{
		litertlm.Image(testImage(t)),
		litertlm.Text("Describe this."),
	})
	if err == nil {
		t.Fatal("SendMulti with image after first turn: want error")
	}
	t.Logf("mid-conversation image error (expected): %v", err)
}

// toneWAV synthesizes a 16 kHz mono PCM16 WAV: a sine tone of the
// given seconds and frequency.
func toneWAV(t *testing.T, seconds int, freq float64) []byte {
	t.Helper()
	const rate = 16000
	n := rate * seconds
	var buf bytes.Buffer
	le := binary.LittleEndian

	dataLen := n * 2
	buf.WriteString("RIFF")
	_ = binary.Write(&buf, le, uint32(36+dataLen))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	_ = binary.Write(&buf, le, uint32(16))
	_ = binary.Write(&buf, le, uint16(1)) // PCM
	_ = binary.Write(&buf, le, uint16(1)) // mono
	_ = binary.Write(&buf, le, uint32(rate))
	_ = binary.Write(&buf, le, uint32(rate*2)) // byte rate
	_ = binary.Write(&buf, le, uint16(2))      // block align
	_ = binary.Write(&buf, le, uint16(16))     // bits
	buf.WriteString("data")
	_ = binary.Write(&buf, le, uint32(dataLen))
	for i := 0; i < n; i++ {
		s := 0.5 * math.Sin(2*math.Pi*freq*float64(i)/rate)
		_ = binary.Write(&buf, le, int16(s*math.MaxInt16))
	}
	return buf.Bytes()
}
