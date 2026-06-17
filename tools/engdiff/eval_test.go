package main

import "testing"

func TestScoreReply(t *testing.T) {
	tests := []struct {
		name  string
		item  evalItem
		reply string
		want  bool
	}{
		{"number exact", evalItem{Kind: "number", Answer: "42"}, "42", true},
		{"number in sentence", evalItem{Kind: "number", Answer: "42"}, "The answer is 42.", true},
		{"number leading zeros", evalItem{Kind: "number", Answer: "7"}, "07", true},
		{"number wrong", evalItem{Kind: "number", Answer: "42"}, "41", false},
		{"number picks first", evalItem{Kind: "number", Answer: "42"}, "42 plus 0 is 42", true},
		{"number none", evalItem{Kind: "number", Answer: "42"}, "I don't know.", false},
		{"contains case-insensitive", evalItem{Kind: "contains", Answer: "Paris"}, "the capital is paris.", true},
		{"contains markdown", evalItem{Kind: "contains", Answer: "Paris"}, "**Paris**", true},
		{"contains wrong", evalItem{Kind: "contains", Answer: "Paris"}, "Lyon", false},
		{"choice bare", evalItem{Kind: "choice", Answer: "B"}, "B", true},
		{"choice with paren", evalItem{Kind: "choice", Answer: "B"}, "B) dolphin", true},
		{"choice lowercase", evalItem{Kind: "choice", Answer: "B"}, "b", true},
		{"choice in sentence", evalItem{Kind: "choice", Answer: "C"}, "The answer is C.", true},
		{"choice wrong", evalItem{Kind: "choice", Answer: "B"}, "A) shark", false},
		{"choice letter inside word ignored", evalItem{Kind: "choice", Answer: "A"}, "Banana", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := scoreReply(tc.item, tc.reply); got != tc.want {
				t.Errorf("scoreReply(%+v, %q) = %v, want %v", tc.item, tc.reply, got, tc.want)
			}
		})
	}
}

func TestLoadEvalSet(t *testing.T) {
	items, err := loadEvalSet("testdata/evalset.json")
	if err != nil {
		t.Fatalf("loadEvalSet: %v", err)
	}
	if len(items) != 60 {
		t.Errorf("len = %d, want 60", len(items))
	}
	for _, it := range items {
		if it.ID == "" || it.Prompt == "" || it.Answer == "" {
			t.Errorf("incomplete item: %+v", it)
		}
		switch it.Kind {
		case "number", "contains", "choice":
		default:
			t.Errorf("item %s has unknown kind %q", it.ID, it.Kind)
		}
	}
}
