package litertlm

import (
	"errors"
	"testing"
)

func TestExperimental_ZeroHandlesAndValidation(t *testing.T) {
	if err := Engine(0).SetMetalResidency(true); err == nil {
		t.Error("expected error on zero Engine SetMetalResidency")
	}

	var nilClient *Client
	if err := nilClient.SetMetalResidency(true); err == nil {
		t.Error("expected error on nil Client SetMetalResidency")
	}
	closedClient := &Client{closed: true}
	if err := closedClient.SetMetalResidency(true); err == nil {
		t.Error("expected error on closed Client SetMetalResidency")
	}

	if err := EngineSettings(0).SetEnableDebugger(true); err == nil {
		t.Error("expected error on zero EngineSettings SetEnableDebugger")
	}

	withStubAPIVersion(t, 1, 0, 0)
	if err := EngineSettings(1).SetEnableDebugger(true); !errors.Is(err, ErrUnimplemented) {
		t.Errorf("SetEnableDebugger on 1.0.0 err = %v, want ErrUnimplemented", err)
	}

	var info SessionDebugInfo
	info.Delete()
	if _, err := info.CaptureDir(); err == nil {
		t.Error("expected error on zero SessionDebugInfo CaptureDir")
	}

	if _, err := Session(0).DebugInfo(); err == nil {
		t.Error("expected error on zero Session DebugInfo")
	}
	if _, err := Session(0).DebugCaptureDir(); err == nil {
		t.Error("expected error on zero Session DebugCaptureDir")
	}
	if _, err := Conversation(0).SessionDebugInfo(); err == nil {
		t.Error("expected error on zero Conversation SessionDebugInfo")
	}
	if _, err := Conversation(0).DebugCaptureDir(); err == nil {
		t.Error("expected error on zero Conversation DebugCaptureDir")
	}

	var nilChat *Chat
	if _, err := nilChat.DebugCaptureDir(); err == nil {
		t.Error("expected error on nil Chat DebugCaptureDir")
	}
	closedChat := &Chat{closed: true}
	if _, err := closedChat.DebugCaptureDir(); err == nil {
		t.Error("expected error on closed Chat DebugCaptureDir")
	}

	cfg := clientConfig{}
	WithDebuggerEnabled(true)(&cfg)
	if cfg.enableDebugger == nil || !*cfg.enableDebugger {
		t.Errorf("WithDebuggerEnabled(true) = %v, want true", cfg.enableDebugger)
	}
}
