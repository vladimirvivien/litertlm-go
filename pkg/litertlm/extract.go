package litertlm

import (
	"fmt"
	"strings"
)

// extractJSON pulls a JSON object or array out of a model's free-form
// response. Tolerates markdown code fences (```json...```), prose
// preambles ("Sure, here it is: {...}"), and trailing commentary.
// Returns the substring containing exactly one balanced top-level
// {...} or [...] (whichever wantArray asks for), ready for
// json.Unmarshal.
//
// The first balanced match wins; subsequent objects in the text are
// ignored. That matches the typical "model returns one block then
// chats about it" failure mode.
func extractJSON(text string, wantArray bool) (string, error) {
	text = stripCodeFences(strings.TrimSpace(text))

	open, closing := byte('{'), byte('}')
	if wantArray {
		open, closing = '[', ']'
	}

	start := strings.IndexByte(text, open)
	if start < 0 {
		return "", fmt.Errorf("no %c found in response", open)
	}

	// Walk the bytes tracking depth and respecting string literals so
	// braces inside strings (`{"msg":"oh {hi"}`) don't fool the counter.
	depth := 0
	inStr := false
	escape := false
	for i := start; i < len(text); i++ {
		c := text[i]
		if escape {
			escape = false
			continue
		}
		if inStr {
			switch c {
			case '\\':
				escape = true
			case '"':
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case open:
			depth++
		case closing:
			depth--
			if depth == 0 {
				return text[start : i+1], nil
			}
		}
	}
	return "", fmt.Errorf("unbalanced %c...%c in response", open, closing)
}

// stripCodeFences removes a leading ```...``` or ```lang...``` wrapper
// when present. Conservative: only strips when the text starts with
// ``` and a closing fence exists somewhere later.
func stripCodeFences(text string) string {
	if !strings.HasPrefix(text, "```") {
		return text
	}
	nl := strings.IndexByte(text, '\n')
	if nl < 0 {
		return text
	}
	inner := text[nl+1:]
	if i := strings.LastIndex(inner, "```"); i >= 0 {
		inner = inner[:i]
	}
	return strings.TrimSpace(inner)
}
