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
	if p.Mime() != "" {
		t.Errorf("Mime() = %q, want empty for text", p.Mime())
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
	if p.Mime() != "" {
		t.Errorf("Mime() = %q, want empty for bare Image", p.Mime())
	}
}

func TestImageWithMime(t *testing.T) {
	p := ImageWithMime([]byte{0xFF}, "image/jpeg")
	if p.Mime() != "image/jpeg" {
		t.Errorf("Mime() = %q, want image/jpeg", p.Mime())
	}
}

func TestImageFromFile(t *testing.T) {
	dir := t.TempDir()
	cases := []struct {
		ext      string
		wantMime string
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
			if p.Mime() != tc.wantMime {
				t.Errorf("Mime() = %q, want %q", p.Mime(), tc.wantMime)
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
	if p.Mime() != "audio/wav" {
		t.Errorf("Mime() = %q, want audio/wav", p.Mime())
	}
}

func TestPartsToInputs_TextOnly(t *testing.T) {
	if !realLibraryLoaded {
		t.Skip("C library not loaded")
	}
	parts := []Part{Text("hello"), Text("world")}
	inputs, err := partsToInputs(parts)
	if err != nil {
		t.Fatalf("partsToInputs failed: %v", err)
	}
	defer func() {
		for _, in := range inputs {
			in.Delete()
		}
	}()
	if len(inputs) != 2 {
		t.Fatalf("len = %d, want 2", len(inputs))
	}
	for i, in := range inputs {
		if in == 0 {
			t.Errorf("inputs[%d] is zero", i)
		}
	}
}

func TestPartsToInputs_ImageEmitsEndMarker(t *testing.T) {
	if !realLibraryLoaded {
		t.Skip("C library not loaded")
	}
	parts := []Part{Image([]byte{0xFF, 0xD8})}
	inputs, err := partsToInputs(parts)
	if err != nil {
		t.Fatalf("partsToInputs failed: %v", err)
	}
	defer func() {
		for _, in := range inputs {
			in.Delete()
		}
	}()
	if len(inputs) != 2 {
		t.Fatalf("len = %d, want 2 (image + end)", len(inputs))
	}
}

func TestPartsToInputs_AudioEmitsEndMarker(t *testing.T) {
	if !realLibraryLoaded {
		t.Skip("C library not loaded")
	}
	parts := []Part{Audio([]byte("riff"))}
	inputs, err := partsToInputs(parts)
	if err != nil {
		t.Fatalf("partsToInputs failed: %v", err)
	}
	defer func() {
		for _, in := range inputs {
			in.Delete()
		}
	}()
	if len(inputs) != 2 {
		t.Fatalf("len = %d, want 2 (audio + end)", len(inputs))
	}
}

func TestPartsToInputs_MixedOrderPreserved(t *testing.T) {
	if !realLibraryLoaded {
		t.Skip("C library not loaded")
	}
	parts := []Part{
		Image([]byte{0xFF}),
		Text("describe"),
		Audio([]byte("riff")),
	}
	inputs, err := partsToInputs(parts)
	if err != nil {
		t.Fatalf("partsToInputs failed: %v", err)
	}
	defer func() {
		for _, in := range inputs {
			in.Delete()
		}
	}()
	if len(inputs) != 5 {
		t.Fatalf("len = %d, want 5", len(inputs))
	}
}

func TestPartsToInputs_Empty(t *testing.T) {
	if !realLibraryLoaded {
		t.Skip("C library not loaded")
	}
	inputs, err := partsToInputs(nil)
	if err != nil {
		t.Fatalf("partsToInputs failed: %v", err)
	}
	defer func() {
		for _, in := range inputs {
			in.Delete()
		}
	}()
	if len(inputs) != 0 {
		t.Errorf("nil parts: len = %d, want 0", len(inputs))
	}
}
