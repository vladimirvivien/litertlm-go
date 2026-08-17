package litertlm_test

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func getLocalLibDir() string {
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

func TestConcurrent_Load(t *testing.T) {
	libDir := getLocalLibDir()
	if libDir == "" {
		t.Skip("skipping concurrent load test: no local prebuilt library directory found")
	}

	var wg sync.WaitGroup
	errs := make(chan error, 20)

	for range 20 {
		wg.Go(func() {
			if err := litertlm.Load(libDir, "cpu", ""); err != nil {
				errs <- err
			}
		})
	}

	wg.Wait()
	close(errs)

	for err := range errs {
		t.Errorf("concurrent Load failed: %v", err)
	}
}

func TestConcurrent_SessionConfig(t *testing.T) {
	libDir := getLocalLibDir()
	if libDir == "" {
		t.Skip("skipping concurrent SessionConfig test: no local prebuilt library directory found")
	}

	if err := litertlm.Load(libDir, "cpu", ""); err != nil {
		t.Fatalf("Load: %v", err)
	}

	var wg sync.WaitGroup
	for range 20 {
		wg.Go(func() {
			cfg, err := litertlm.NewSessionConfig()
			if err != nil {
				t.Errorf("NewSessionConfig: %v", err)
				return
			}
			defer cfg.Delete()

			cfg.SetMaxOutputTokens(64)
			cfg.SetApplyPromptTemplate(true)
			_ = cfg.SetLoraPath("nonexistent.lora")
			_ = cfg.SetAudioLoraPath("nonexistent_audio.lora")
		})
	}
	wg.Wait()
}

func TestConcurrent_ClientLifecycle(t *testing.T) {
	libDir := getLocalLibDir()
	modelPath := os.Getenv("LITERTLM_MODEL")
	if modelPath == "" {
		defaultModel := filepath.Join(os.Getenv("USERPROFILE"), "models", "gemma3-1b-it-int4.litertlm")
		if _, err := os.Stat(defaultModel); err == nil {
			modelPath = defaultModel
		}
	}
	if libDir == "" || modelPath == "" {
		t.Skip("skipping live concurrent Client test: library or model path not found")
	}

	ctx := context.Background()
	client, err := litertlm.New(ctx,
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
		litertlm.WithBackend("cpu"),
	)
	if err != nil {
		t.Fatalf("litertlm.New: %v", err)
	}
	defer func() { _ = client.Close() }()

	var wg sync.WaitGroup
	for i := range 10 {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			tokens, err := client.Tokenize("Hello world from concurrent goroutine")
			if err != nil {
				t.Errorf("Tokenize failed in goroutine %d: %v", id, err)
				return
			}
			if len(tokens) == 0 {
				t.Errorf("expected non-empty tokens in goroutine %d", id)
			}
		}(i)
	}
	wg.Wait()
}
