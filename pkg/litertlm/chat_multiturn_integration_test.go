package litertlm_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

// Multi-turn integration tests against whatever model
// LITERTLM_TEST_MODEL points at. On Gemma 4 these confirm baseline
// turn-N behavior; on Qwen3 they reproduce the "second SendMessage
// fails" failure mode. Same harness for both.
//
// Tests exercise the low-level Conversation API directly so any
// failure isolates to the C-side state machine, not the Chat wrapper.

func multiturnUserMessage(t *testing.T, content string) string {
	t.Helper()
	b, err := json.Marshal(map[string]string{"role": "user", "content": content})
	if err != nil {
		t.Fatalf("marshal user message: %v", err)
	}
	return string(b)
}

// TestMultiTurn_TwoTextTurns sends two consecutive user messages on
// the same Conversation. A pass means the model handles multi-turn
// state correctly; a fail surfaces the reported Qwen3 bug.
func TestMultiTurn_TwoTextTurns(t *testing.T) {
	conv := buildIntegrationConversation(t)

	turn1 := multiturnUserMessage(t, "Reply with exactly one word.")
	reply1, err := conv.SendMessage(turn1, "", litertlm.OptionalArgs(0))
	if err != nil {
		t.Fatalf("turn 1 SendMessage: %v", err)
	}
	if reply1 == "" {
		t.Fatal("turn 1 reply was empty")
	}
	t.Logf("turn-1 reply: %s", reply1)

	turn2 := multiturnUserMessage(t, "What did I just ask?")
	reply2, err := conv.SendMessage(turn2, "", litertlm.OptionalArgs(0))
	if err != nil {
		t.Fatalf("turn 2 SendMessage: %v", err)
	}
	if reply2 == "" {
		t.Fatal("turn 2 reply was empty")
	}
	t.Logf("turn-2 reply: %s", reply2)
}

// TestMultiTurn_RenderTurn2_AfterTurn1Sent is the bug-localization
// diagnostic. Turn 1 is sent so the Conversation absorbs state;
// turn 2 is rendered only, never decoded.
//
// If this test passes (RenderMessage succeeds) but
// TestMultiTurn_TwoTextTurns fails, the bug is in SendMessage's
// decode / state advance — i.e., at or below the C-side decode
// pipeline. If both fail at this test, the bug is in template encode
// and shows up before any decode happens.
func TestMultiTurn_RenderTurn2_AfterTurn1Sent(t *testing.T) {
	conv := buildIntegrationConversation(t)

	turn1 := multiturnUserMessage(t, "Reply with exactly one word.")
	if _, err := conv.SendMessage(turn1, "", litertlm.OptionalArgs(0)); err != nil {
		t.Fatalf("turn 1 SendMessage: %v", err)
	}

	turn2 := multiturnUserMessage(t, "What did I just ask?")
	rendered, err := conv.RenderMessage(turn2)
	if err != nil {
		t.Fatalf("turn 2 RenderMessage: %v (bug is template/encode-side)", err)
	}
	if rendered == "" {
		t.Fatal("turn 2 RenderMessage returned empty string")
	}
	// The rendered envelope is the diagnostic artifact. Dump it so
	// the test log carries enough context to compare against the
	// model's documented chat template.
	t.Logf("turn-2 rendered envelope:\n%s", rendered)

	if !strings.Contains(rendered, "What did I just ask?") {
		t.Errorf("rendered envelope missing turn-2 user content")
	}
}

// TestMultiTurn_ToolResultTurn sends a normal user message on turn 1,
// then a synthesized tool-role envelope on turn 2. No tools are
// registered in the Conversation config — the test probes whether
// the C side accepts a non-user role envelope on turn 2 at all.
func TestMultiTurn_ToolResultTurn(t *testing.T) {
	conv := buildIntegrationConversation(t)

	turn1 := multiturnUserMessage(t, "Pretend you called a tool named demo with no args.")
	if _, err := conv.SendMessage(turn1, "", litertlm.OptionalArgs(0)); err != nil {
		t.Fatalf("turn 1 SendMessage: %v", err)
	}

	toolMsg, err := json.Marshal(map[string]any{
		"role": "tool",
		"content": []map[string]any{
			{"name": "demo", "response": map[string]any{"value": 42}},
		},
	})
	if err != nil {
		t.Fatalf("marshal tool result: %v", err)
	}
	reply, err := conv.SendMessage(string(toolMsg), "", litertlm.OptionalArgs(0))
	if err != nil {
		t.Fatalf("tool-result SendMessage: %v", err)
	}
	t.Logf("tool-result reply: %s", reply)
}
