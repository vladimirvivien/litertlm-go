package litertgo

import (
	"errors"
	"testing"

	"github.com/vladimirvivien/litert-go/lm"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func TestMessageText(t *testing.T) {
	tests := []struct {
		name    string
		json    string
		want    string
		wantErr bool
	}{
		{
			name: "string content",
			json: `{"role":"user","content":"hello"}`,
			want: "hello",
		},
		{
			name: "content array",
			json: `{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}`,
			want: "ab",
		},
		{
			name: "missing role defaults to user",
			json: `{"content":"hi"}`,
			want: "hi",
		},
		{
			name:    "tool role unsupported",
			json:    `{"role":"tool","content":[{"name":"f","response":{}}]}`,
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
			got, err := messageText(tc.json)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("messageText(%s): want error, got %q", tc.json, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("messageText(%s): %v", tc.json, err)
			}
			if got != tc.want {
				t.Errorf("messageText(%s) = %q, want %q", tc.json, got, tc.want)
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

func TestAssistantEnvelope_RoundTripsText(t *testing.T) {
	env, err := assistantEnvelope(`reply with "quotes" and
newline`)
	if err != nil {
		t.Fatalf("assistantEnvelope: %v", err)
	}
	// The envelope must parse back through the adapter's own message
	// reader shape (same content-array schema the Chat parser uses).
	got, err := messageText(env)
	if err == nil {
		t.Fatalf("messageText on assistant envelope: want role error, got %q", got)
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
		{"tools", litertlm.ConversationSetup{ToolsJSON: `[{"type":"function"}]`}},
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
