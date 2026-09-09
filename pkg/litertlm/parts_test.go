package litertlm

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestText(t *testing.T) {
	p := Text("hello")
	if !p.IsText() {
		t.Errorf("IsText() = false")
	}
	if p.text != "hello" {
		t.Errorf("text = %q, want hello", p.text)
	}
	if p.MIME() != "" {
		t.Errorf("MIME() = %q, want empty for text", p.MIME())
	}
}

func TestImage(t *testing.T) {
	jpeg := []byte{0xFF, 0xD8, 0xFF, 0xE0}
	p := Image(jpeg)
	if !p.IsImage() {
		t.Errorf("IsImage() = false")
	}
	if !bytes.Equal(p.data, jpeg) {
		t.Errorf("data mismatch")
	}
	if p.MIME() != "" {
		t.Errorf("MIME() = %q, want empty for bare Image", p.MIME())
	}
}

func TestImageWithMIME(t *testing.T) {
	p := ImageWithMIME([]byte{0xFF}, "image/jpeg")
	if p.MIME() != "image/jpeg" {
		t.Errorf("MIME() = %q, want image/jpeg", p.MIME())
	}
}

func TestImageFromFile(t *testing.T) {
	dir := t.TempDir()
	cases := []struct {
		ext      string
		wantMIME string
	}{
		{".jpg", "image/jpeg"},
		{".JPG", "image/jpeg"},
		{".png", "image/png"},
		{".webp", "image/webp"},
		{".xyz", ""}, // unknown — empty mime, bytes still wrapped
	}
	for _, tc := range cases {
		t.Run(tc.ext, func(t *testing.T) {
			path := filepath.Join(dir, "fixture"+tc.ext)
			payload := []byte{0xAB, 0xCD, 0xEF}
			if err := os.WriteFile(path, payload, 0o600); err != nil {
				t.Fatalf("write: %v", err)
			}
			p, err := ImageFromFile(path)
			if err != nil {
				t.Fatalf("ImageFromFile: %v", err)
			}
			if p.MIME() != tc.wantMIME {
				t.Errorf("MIME() = %q, want %q", p.MIME(), tc.wantMIME)
			}
			if !bytes.Equal(p.data, payload) {
				t.Errorf("payload mismatch")
			}
		})
	}
}

func TestImageFromFile_NotFound(t *testing.T) {
	_, err := ImageFromFile("/nonexistent/path/no.jpg")
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestAudio(t *testing.T) {
	wav := []byte("RIFF\x00\x00\x00\x00WAVE")
	p := Audio(wav)
	if !p.IsAudio() {
		t.Errorf("IsAudio() = false")
	}
	if !bytes.Equal(p.data, wav) {
		t.Errorf("data mismatch")
	}
}

func TestAudioFromFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "clip.wav")
	if err := os.WriteFile(path, []byte("riff"), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	p, err := AudioFromFile(path)
	if err != nil {
		t.Fatalf("AudioFromFile: %v", err)
	}
	if p.MIME() != "audio/wav" {
		t.Errorf("MIME() = %q, want audio/wav", p.MIME())
	}
}
