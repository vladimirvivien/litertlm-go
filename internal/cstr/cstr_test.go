package cstr

import "testing"

func TestBytePtrRoundTrip(t *testing.T) {
	tests := []struct {
		name string
		s    string
	}{
		{"ascii", "hello, world"},
		{"unicode", "café — naïve résumé"},
		{"empty", ""},
		{"single_char", "a"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ptr, err := BytePtrFromString(tt.s)
			if err != nil {
				t.Fatalf("BytePtrFromString(%q): unexpected error: %v", tt.s, err)
			}
			got := BytePtrToString(ptr)
			if got != tt.s {
				t.Errorf("round trip = %q, want %q", got, tt.s)
			}
		})
	}
}

func TestBytePtrFromStringNullByte(t *testing.T) {
	_, err := BytePtrFromString("abc\x00def")
	if err == nil {
		t.Fatal("expected error for string containing null byte, got nil")
	}
}
