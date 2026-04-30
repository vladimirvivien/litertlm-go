//go:build linux || freebsd || darwin

package utils

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
	// unix.BytePtrFromString rejects strings containing NUL bytes
	// because the result would be ambiguous when re-read as
	// null-terminated. Confirm our wrapper preserves that behavior.
	_, err := BytePtrFromString("abc\x00def")
	if err == nil {
		t.Fatal("expected error for string containing null byte, got nil")
	}
}
