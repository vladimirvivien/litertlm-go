package litertlm

import (
	"testing"
	"unsafe"
)

// InputData layout must stay 24 bytes to match the C struct. The
// compile-time check in input.go would catch a regression too, but a
// runtime test surfaces it as a clean test failure with the actual size.
func TestInputDataLayout(t *testing.T) {
	if got, want := unsafe.Sizeof(InputData{}), uintptr(24); got != want {
		t.Fatalf("sizeof(InputData) = %d, want %d", got, want)
	}
}

func TestNewTextInput(t *testing.T) {
	tests := []struct {
		name     string
		s        []byte
		wantSize uintptr
		wantNil  bool
	}{
		{"non-empty", []byte("hello"), 5, false},
		{"empty", []byte{}, 0, true},
		{"nil", nil, 0, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := NewTextInput(tt.s)
			if in.Type != InputText {
				t.Errorf("Type = %d, want InputText (%d)", in.Type, InputText)
			}
			if in.Size != tt.wantSize {
				t.Errorf("Size = %d, want %d", in.Size, tt.wantSize)
			}
			if (in.Data == nil) != tt.wantNil {
				t.Errorf("Data nilness = %v, want %v", in.Data == nil, tt.wantNil)
			}
			if !tt.wantNil && in.Data != unsafe.Pointer(&tt.s[0]) {
				t.Errorf("Data should point to first byte of input")
			}
		})
	}
}

func TestNewTextInputString(t *testing.T) {
	t.Run("non-empty", func(t *testing.T) {
		s := "hello"
		in := NewTextInputString(s)
		if in.Type != InputText {
			t.Errorf("Type = %d, want InputText", in.Type)
		}
		if in.Size != uintptr(len(s)) {
			t.Errorf("Size = %d, want %d", in.Size, len(s))
		}
		if in.Data != unsafe.Pointer(unsafe.StringData(s)) {
			t.Errorf("Data should point to string backing bytes")
		}
	})
	t.Run("empty", func(t *testing.T) {
		in := NewTextInputString("")
		if in.Size != 0 || in.Data != nil {
			t.Errorf("empty string: got Size=%d Data=%v, want 0 / nil", in.Size, in.Data)
		}
	})
}

func TestNewBinaryInput(t *testing.T) {
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
			in := NewBinaryInput(tt.typ, b)
			if in.Type != tt.typ {
				t.Errorf("Type = %d, want %d", in.Type, tt.typ)
			}
			if in.Size != uintptr(len(b)) {
				t.Errorf("Size = %d, want %d", in.Size, len(b))
			}
			if in.Data != unsafe.Pointer(&b[0]) {
				t.Errorf("Data should point to first byte")
			}
		})
	}
	t.Run("empty_payload", func(t *testing.T) {
		in := NewBinaryInput(InputImage, nil)
		if in.Type != InputImage || in.Size != 0 || in.Data != nil {
			t.Errorf("nil payload: got %+v", in)
		}
	})
}
