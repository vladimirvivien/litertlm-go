package litertlm

import (
	"testing"
)

func TestNewTextInput(t *testing.T) {
	if !realLibraryLoaded {
		t.Skip("C library not loaded")
	}
	tests := []struct {
		name string
		s    []byte
	}{
		{"non-empty", []byte("hello")},
		{"empty", []byte{}},
		{"nil", nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in, err := NewTextInput(tt.s)
			if err != nil {
				t.Fatalf("NewTextInput failed: %v", err)
			}
			defer in.Delete()
			if in == 0 {
				t.Errorf("expected non-zero handle")
			}
		})
	}
}

func TestNewTextInputString(t *testing.T) {
	if !realLibraryLoaded {
		t.Skip("C library not loaded")
	}
	t.Run("non-empty", func(t *testing.T) {
		s := "hello"
		in, err := NewTextInputString(s)
		if err != nil {
			t.Fatalf("NewTextInputString failed: %v", err)
		}
		defer in.Delete()
		if in == 0 {
			t.Errorf("expected non-zero handle")
		}
	})
	t.Run("empty", func(t *testing.T) {
		in, err := NewTextInputString("")
		if err != nil {
			t.Fatalf("NewTextInputString failed: %v", err)
		}
		defer in.Delete()
		if in == 0 {
			t.Errorf("expected non-zero handle")
		}
	})
}

func TestNewBinaryInput(t *testing.T) {
	if !realLibraryLoaded {
		t.Skip("C library not loaded")
	}
	tests := []struct {
		name string
		typ  InputDataType
	}{
		{"image", InputImage},
		{"image_end", InputImageEnd},
		{"audio", InputAudio},
		{"audio_end", InputAudioEnd},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := []byte{0x01, 0x02, 0x03}
			in, err := NewBinaryInput(tt.typ, b)
			if err != nil {
				t.Fatalf("NewBinaryInput failed: %v", err)
			}
			defer in.Delete()
			if in == 0 {
				t.Errorf("expected non-zero handle")
			}
		})
	}
	t.Run("empty_payload", func(t *testing.T) {
		in, err := NewBinaryInput(InputImage, nil)
		if err != nil {
			t.Fatalf("NewBinaryInput failed: %v", err)
		}
		defer in.Delete()
		if in == 0 {
			t.Errorf("expected non-zero handle")
		}
	})
}
