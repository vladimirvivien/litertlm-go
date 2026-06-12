package litertgo_test

import (
	"context"
	"errors"
	"os"
	"testing"

	"github.com/vladimirvivien/litert-go/lm"
	"github.com/vladimirvivien/litertlm-go/backend/litertgo"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

var weatherSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"city": map[string]any{"type": "string", "description": "City name."},
	},
	"required": []string{"city"},
}

// TestGoBackend_RawToolRound mirrors the C++ probe flow: raw tool,
// manual SendToolResult. The final reply is pinned to the C++
// engine's greedy output for this exact round on gemma-4-E2B-it.
func TestGoBackend_RawToolRound(t *testing.T) {
	c := newGemma4Client(t)

	weather := litertlm.NewRawTool("get_weather", "Get current weather for a city.", weatherSchema)
	chat, err := c.NewChat(context.Background(),
		litertlm.WithSystemPrompt("You are a helpful assistant."),
		litertlm.WithTool(weather))
	if err != nil {
		t.Fatalf("NewChat: %v", err)
	}
	defer chat.Close()

	reply, err := chat.Send(context.Background(), "What is the weather in Paris?")
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	if !reply.HasToolCalls() {
		t.Fatalf("expected tool call, got text: %q", reply.Text())
	}
	call := reply.ToolCalls()[0]
	if call.Function.Name != "get_weather" || call.Function.Arguments["city"] != "Paris" {
		t.Fatalf("call = %+v, want get_weather(city=Paris)", call)
	}

	final, err := chat.SendToolResult(context.Background(), "get_weather",
		map[string]any{"temp_c": 21, "sky": "clear"})
	if err != nil {
		t.Fatalf("SendToolResult: %v", err)
	}
	const anchor = "The weather in Paris is clear with a temperature of 21°C."
	if final.Text() != anchor {
		t.Errorf("final = %q, want C++ anchor %q", final.Text(), anchor)
	}
}

// TestGoBackend_ManagedToolDispatch exercises the auto-dispatch loop:
// a typed ManagedTool invoked by the framework inside one Chat.Send.
func TestGoBackend_ManagedToolDispatch(t *testing.T) {
	c := newGemma4Client(t)

	type weatherIn struct {
		City string `json:"city" description:"City name."`
	}
	type weatherOut struct {
		TempC int    `json:"temp_c"`
		Sky   string `json:"sky"`
	}
	var invokedCity string
	tool, err := litertlm.RegisterTool(c, "get_weather", "Get current weather for a city.",
		func(ctx context.Context, in weatherIn) (weatherOut, error) {
			invokedCity = in.City
			return weatherOut{TempC: 21, Sky: "clear"}, nil
		})
	if err != nil {
		t.Fatalf("RegisterTool: %v", err)
	}

	chat, err := c.NewChat(context.Background(),
		litertlm.WithSystemPrompt("You are a helpful assistant."),
		litertlm.WithTool(tool))
	if err != nil {
		t.Fatalf("NewChat: %v", err)
	}
	defer chat.Close()

	reply, err := chat.Send(context.Background(), "What is the weather in Paris?")
	if err != nil {
		t.Fatalf("Send: %v", err)
	}
	if invokedCity != "Paris" {
		t.Errorf("tool invoked with city %q, want Paris", invokedCity)
	}
	if reply.HasToolCalls() || reply.Text() == "" {
		t.Errorf("want final text reply, got %q (calls=%v)", reply.Text(), reply.ToolCalls())
	}
	t.Logf("final: %q", reply.Text())
}

// TestGoBackend_ToolsOnUnsupportedFamily verifies NewChat fails fast
// with tools on a family litert-go has no tool syntax for.
func TestGoBackend_ToolsOnUnsupportedFamily(t *testing.T) {
	if testing.Short() {
		t.Skip("short mode")
	}
	libDir := os.Getenv("LITERT_LIB")
	modelPath := os.Getenv("LITERTLM_TEST_MODEL") // gemma3-270m in the standard run
	if libDir == "" || modelPath == "" {
		t.Skip("LITERT_LIB / LITERTLM_TEST_MODEL not set")
	}
	b, err := litertgo.Open(context.Background(), modelPath, lm.WithLibDir(libDir))
	if err != nil {
		t.Fatalf("litertgo.Open: %v", err)
	}
	c, err := litertlm.New(context.Background(), litertlm.WithEngineBackend(b))
	if err != nil {
		b.Close()
		t.Fatalf("litertlm.New: %v", err)
	}
	defer c.Close()

	weather := litertlm.NewRawTool("get_weather", "Get current weather.", weatherSchema)
	_, err = c.NewChat(context.Background(), litertlm.WithTool(weather))
	if err == nil {
		t.Fatal("NewChat with tools on non-tool family: want error")
	}
	if errors.Is(err, litertgo.ErrUnsupported) {
		t.Logf("got ErrUnsupported as expected: %v", err)
	}
}
