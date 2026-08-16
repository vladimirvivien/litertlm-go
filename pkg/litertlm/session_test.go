package litertlm

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSessionConfig_NilHandleErrors(t *testing.T) {
	var cfg SessionConfig
	if err := cfg.SetLoraPath("dummy.lora"); err == nil {
		t.Error("expected error for zero SessionConfig.SetLoraPath, got nil")
	} else if !strings.Contains(err.Error(), "invalid session config") {
		t.Errorf("unexpected error message: %v", err)
	}

	if err := cfg.SetAudioLoraPath("dummy_audio.lora"); err == nil {
		t.Error("expected error for zero SessionConfig.SetAudioLoraPath, got nil")
	} else if !strings.Contains(err.Error(), "invalid session config") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestConversationConfig_NilHandleErrors(t *testing.T) {
	var cfg ConversationConfig
	if err := cfg.SetStreamToolCalls(true, "thought"); err == nil {
		t.Error("expected error for zero ConversationConfig.SetStreamToolCalls, got nil")
	} else if !strings.Contains(err.Error(), "invalid conversation config") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func getTestLibDir() string {
	libDir := os.Getenv("LITERTLM_LIB")
	if libDir != "" {
		return libDir
	}
	defaultInclude := filepath.Join(os.Getenv("USERPROFILE"), "include", "litertlm", "lib")
	if _, err := os.Stat(defaultInclude); err == nil {
		return defaultInclude
	}
	return ""
}

func TestLive_SessionConfig_LoRAPaths(t *testing.T) {
	libDir := getTestLibDir()
	if libDir == "" {
		t.Skip("skipping live test: no local prebuilt library directory found")
	}

	if err := Load(libDir, "cpu", ""); err != nil {
		t.Fatalf("Load: %v", err)
	}

	sessCfg, err := NewSessionConfig()
	if err != nil {
		t.Fatalf("NewSessionConfig: %v", err)
	}
	defer sessCfg.Delete()

	// Calling with empty or path string exercises the C API binding
	if err := sessCfg.SetLoraPath("nonexistent.lora"); err != nil {
		// Non-zero return from C side for invalid path is expected and demonstrates invocation
		t.Logf("SetLoraPath returned expected error on non-existent path: %v", err)
	}

	if err := sessCfg.SetAudioLoraPath("nonexistent_audio.lora"); err != nil {
		t.Logf("SetAudioLoraPath returned expected error on non-existent path: %v", err)
	}
}

func TestLive_ConversationConfig_StreamToolCalls(t *testing.T) {
	libDir := getTestLibDir()
	if libDir == "" {
		t.Skip("skipping live test: no local prebuilt library directory found")
	}

	if err := Load(libDir, "cpu", ""); err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Create conversation config via FFI create
	convCfgHandle, err := callForHandle[ConversationConfig](conversationConfigCreateFunc, "conversation_config_create")
	if err != nil {
		t.Fatalf("conversation_config_create: %v", err)
	}
	defer convCfgHandle.Delete()

	// Test SetStreamToolCalls on live handle
	if err := convCfgHandle.SetStreamToolCalls(true, "thought_channel"); err != nil {
		t.Fatalf("SetStreamToolCalls failed on valid handle: %v", err)
	}

	if err := convCfgHandle.SetStreamToolCalls(false, ""); err != nil {
		t.Fatalf("SetStreamToolCalls(false, \"\") failed: %v", err)
	}
}
