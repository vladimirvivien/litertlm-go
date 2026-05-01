package litertlm

import "testing"

func TestChunk_ZeroValue(t *testing.T) {
	var c Chunk
	if c.Text != "" {
		t.Errorf("zero Text = %q, want empty", c.Text)
	}
	if c.Final {
		t.Errorf("zero Final = true, want false")
	}
}

func TestChunk_ValueSemantics(t *testing.T) {
	a := Chunk{Text: "hello", Final: true}
	b := a
	b.Text = "world"
	if a.Text != "hello" {
		t.Errorf("mutation through b leaked into a: a.Text = %q", a.Text)
	}
}
