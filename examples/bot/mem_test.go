package main

import (
	"strings"
	"testing"
	"time"
)

func TestMemoryRoundtrip(t *testing.T) {
	in := []turn{
		{role: "user", ts: time.Date(2026, 5, 10, 14, 30, 12, 0, time.UTC), body: "explain why the sky is blue"},
		{role: "bot", ts: time.Date(2026, 5, 10, 14, 30, 15, 0, time.UTC), body: "Rayleigh scattering.\nShorter wavelengths scatter more."},
		{role: "user", ts: time.Date(2026, 5, 10, 14, 31, 5, 0, time.UTC), body: "show me a code block:\n```python\nprint('hi')\n```"},
		{role: "bot", ts: time.Date(2026, 5, 10, 14, 31, 6, 0, time.UTC), body: "ok\n"},
	}
	rendered := renderTurns(in)
	out, err := parseMemory(rendered)
	if err != nil {
		t.Fatalf("parseMemory: %v\n--- rendered ---\n%s", err, rendered)
	}
	if len(out) != len(in) {
		t.Fatalf("got %d turns, want %d", len(out), len(in))
	}
	for i := range in {
		if out[i].role != in[i].role {
			t.Errorf("turn %d role = %q, want %q", i, out[i].role, in[i].role)
		}
		if !out[i].ts.Equal(in[i].ts) {
			t.Errorf("turn %d ts = %v, want %v", i, out[i].ts, in[i].ts)
		}
		want := strings.TrimRight(in[i].body, "\n")
		got := strings.TrimRight(out[i].body, "\n")
		if got != want {
			t.Errorf("turn %d body mismatch\ngot:  %q\nwant: %q", i, got, want)
		}
	}
}

func TestMemoryParseRejectsMissingFence(t *testing.T) {
	bad := "> user 2026-05-10T14:30:12Z````\nhello\n"
	if _, err := parseMemory(bad); err == nil {
		t.Fatal("expected error on missing closing fence, got nil")
	}
}

func TestMemoryParseRejectsBadHeader(t *testing.T) {
	bad := "not a header\nstuff\n````\n"
	if _, err := parseMemory(bad); err == nil {
		t.Fatal("expected error on non-header line, got nil")
	}
}

func TestMemoryParseEmpty(t *testing.T) {
	turns, err := parseMemory("")
	if err != nil {
		t.Fatalf("empty input: %v", err)
	}
	if len(turns) != 0 {
		t.Fatalf("empty input: got %d turns, want 0", len(turns))
	}
}

func TestIsExitCommand(t *testing.T) {
	cases := []struct {
		in   string
		want bool
	}{
		{"exit", true},
		{"/exit", true},
		{"EXIT", true},
		{"Exit", true},
		{"  exit  ", true},
		{"  /exit  ", true},
		{"exiting", false},
		{"please exit", false},
		{"", false},
		{"quit", false},
		{"/quit", false},
	}
	for _, c := range cases {
		if got := isExitCommand(c.in); got != c.want {
			t.Errorf("isExitCommand(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}
