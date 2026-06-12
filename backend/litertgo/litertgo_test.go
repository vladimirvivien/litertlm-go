package litertgo

import (
	"errors"
	"reflect"
	"testing"

	"github.com/vladimirvivien/litert-go/lm"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func TestDecodeMessage(t *testing.T) {
	tests := []struct {
		name    string
		json    string
		want    message
		wantErr bool
	}{
		{
			name: "string content",
			json: `{"role":"user","content":"hello"}`,
			want: message{Role: "user", Text: "hello"},
		},
		{
			name: "content array",
			json: `{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}`,
			want: message{Role: "user", Text: "ab"},
		},
		{
			name: "missing role defaults to user",
			json: `{"content":"hi"}`,
			want: message{Role: "user", Text: "hi"},
		},
		{
			name: "tool results",
			json: `{"role":"tool","content":[{"name":"get_weather","response":{"temp_c":21}}]}`,
			want: message{Role: "tool", Results: []lm.ToolResult{
				{Name: "get_weather", Response: map[string]any{"temp_c": float64(21)}},
			}},
		},
		{
			name:    "assistant role unsupported",
			json:    `{"role":"assistant","content":"hi"}`,
			wantErr: true,
		},
		{
			name:    "image part unsupported",
			json:    `{"role":"user","content":[{"type":"image","blob":"..."}]}`,
			wantErr: true,
		},
		{
			name:    "not json",
			json:    `nope`,
			wantErr: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := decodeMessage(tc.json)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("decodeMessage(%s): want error, got %+v", tc.json, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("decodeMessage(%s): %v", tc.json, err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("decodeMessage(%s) = %+v, want %+v", tc.json, got, tc.want)
			}
		})
	}
}

func TestSystemText(t *testing.T) {
	if got, err := systemText(`"be brief"`); err != nil || got != "be brief" {
		t.Errorf("systemText = %q, %v; want \"be brief\", nil", got, err)
	}
	if got, err := systemText(""); err != nil || got != "" {
		t.Errorf("systemText(\"\") = %q, %v; want \"\", nil", got, err)
	}
	if _, err := systemText(`{not json`); err == nil {
		t.Error("systemText(invalid): want error")
	}
}

func TestReplyEnvelope(t *testing.T) {
	env, err := replyEnvelope("It is sunny.", nil)
	if err != nil {
		t.Fatalf("replyEnvelope: %v", err)
	}
	if want := `{"content":[{"text":"It is sunny.","type":"text"}],"role":"assistant"}`; env != want {
		t.Errorf("text envelope = %s, want %s", env, want)
	}

	env, err = replyEnvelope("", []lm.ToolCall{{Name: "get_weather", Args: map[string]any{"city": "Paris"}}})
	if err != nil {
		t.Fatalf("replyEnvelope(calls): %v", err)
	}
	if want := `{"role":"assistant","tool_calls":[{"function":{"arguments":{"city":"Paris"},"name":"get_weather"},"type":"function"}]}`; env != want {
		t.Errorf("call envelope = %s, want %s", env, want)
	}
}

func TestGenOptions(t *testing.T) {
	tests := []struct {
		name    string
		max     int
		sampler *litertlm.SamplerParams
		system  string
		want    lm.GenOptions
	}{
		{
			name: "nil sampler is greedy",
			max:  64,
			want: lm.GenOptions{MaxTokens: 64},
		},
		{
			name:    "greedy type is greedy",
			sampler: &litertlm.SamplerParams{Type: litertlm.SamplerGreedy, Temperature: 0.9, TopK: 40},
			want:    lm.GenOptions{MaxTokens: defaultMaxOutputTokens},
		},
		{
			name:    "topk maps fields",
			max:     64,
			sampler: &litertlm.SamplerParams{Type: litertlm.SamplerTopK, Temperature: 0.7, TopK: 40, Seed: 11},
			want:    lm.GenOptions{MaxTokens: 64, Temp: 0.7, TopK: 40, Seed: 11},
		},
		{
			name:    "topp maps topp too",
			max:     64,
			sampler: &litertlm.SamplerParams{Type: litertlm.SamplerTopP, Temperature: 0.7, TopK: 40, TopP: 0.95, Seed: 11},
			want:    lm.GenOptions{MaxTokens: 64, Temp: 0.7, TopK: 40, TopP: 0.95, Seed: 11},
		},
		{
			name:   "unset max gets default cap",
			system: "be brief",
			want:   lm.GenOptions{MaxTokens: defaultMaxOutputTokens, System: "be brief"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := genOptions(tc.max, tc.sampler, tc.system)
			if got != tc.want {
				t.Errorf("genOptions = %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestNewChatTransport_UnsupportedSetups(t *testing.T) {
	b := &Backend{}
	tests := []struct {
		name  string
		setup litertlm.ConversationSetup
	}{
		{"initial messages", litertlm.ConversationSetup{MessagesJSON: `[]`}},
		{"constrained decoding", litertlm.ConversationSetup{ConstrainedDecoding: true}},
		{"extra context", litertlm.ConversationSetup{ExtraContextJSON: `{"k":"v"}`}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := b.NewChatTransport(tc.setup)
			if !errors.Is(err, ErrUnsupported) {
				t.Errorf("NewChatTransport(%s) err = %v, want ErrUnsupported", tc.name, err)
			}
		})
	}
}
