package litertlm

import "testing"

func TestExtractJSON_Object(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"pure object", `{"a":1}`, `{"a":1}`},
		{"trimmed whitespace", "   {\"a\":1}\n", `{"a":1}`},
		{"fenced json", "```json\n{\"a\":1}\n```", `{"a":1}`},
		{"fenced no lang", "```\n{\"a\":1}\n```", `{"a":1}`},
		{"preamble", `Sure: {"a":1}`, `{"a":1}`},
		{"trailing prose", `{"a":1} - that's the answer!`, `{"a":1}`},
		{"preamble and trailing", `Here you go: {"a":1}. Hope that helps!`, `{"a":1}`},
		{"nested", `{"a":{"b":1}}`, `{"a":{"b":1}}`},
		{"string with brace", `{"msg":"oh {hi"}`, `{"msg":"oh {hi"}`},
		{"string with escaped quote", `{"msg":"a \"b\" c"}`, `{"msg":"a \"b\" c"}`},
		{"string with escaped close brace", `{"msg":"\"close: }"}`, `{"msg":"\"close: }"}`},
		{"first balanced wins", `{"a":1} then {"b":2}`, `{"a":1}`},
		{"multiline pretty", "{\n  \"a\": 1,\n  \"b\": 2\n}", "{\n  \"a\": 1,\n  \"b\": 2\n}"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := extractJSON(tt.input, false)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Errorf("got  = %q\nwant = %q", got, tt.want)
			}
		})
	}
}

func TestExtractJSON_Array(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"pure array", `[1,2,3]`, `[1,2,3]`},
		{"fenced array", "```json\n[1,2,3]\n```", `[1,2,3]`},
		{"array of objects", `[{"a":1},{"a":2}]`, `[{"a":1},{"a":2}]`},
		{"preamble + array", `Result: [1,2,3]`, `[1,2,3]`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := extractJSON(tt.input, true)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Errorf("got  = %q\nwant = %q", got, tt.want)
			}
		})
	}
}

func TestExtractJSON_Errors(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantArray bool
	}{
		{"no JSON object", `just words`, false},
		{"no JSON array", `just words`, true},
		{"unbalanced object", `{"a":1`, false},
		{"unbalanced array", `[1,2,3`, true},
		{"object when array wanted", `{"a":1}`, true},
		{"empty input", ``, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := extractJSON(tt.input, tt.wantArray)
			if err == nil {
				t.Errorf("expected error, got nil")
			}
		})
	}
}

func TestStripCodeFences(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"plain text", "plain text"},
		{"```json\n{}\n```", "{}"},
		{"```\n[1]\n```", "[1]"},
		{"```yaml\nfoo: bar\n```", "foo: bar"},
		{"```\nno closing fence\n", "no closing fence"}, // best effort
		{"   ```json\n{}\n```", "   ```json\n{}\n```"},  // doesn't strip when prefix isn't ```
	}
	for _, c := range cases {
		if got := stripCodeFences(c.in); got != c.want {
			t.Errorf("stripCodeFences(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}
