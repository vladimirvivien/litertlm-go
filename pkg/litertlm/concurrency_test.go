package litertlm_test

import (
	"context"
	"os"
	"sync"
	"testing"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func getLocalLibDir() string {
	for _, env := range []string{"LITERTLM_TEST_LIB", "LITERTLM_LIB"} {
		if libDir := os.Getenv(env); libDir != "" {
			return libDir
		}
	}
	return ""
}

func TestConcurrent_Load(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test skipped in -short mode")
	}
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
	if testing.Short() {
		t.Skip("integration test skipped in -short mode")
	}
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

			if err := cfg.SetMaxOutputTokens(64); err != nil {
				t.Errorf("SetMaxOutputTokens: %v", err)
			}
			if err := cfg.SetApplyPromptTemplate(true); err != nil {
				t.Errorf("SetApplyPromptTemplate: %v", err)
			}
			_ = cfg.SetLoRAPath("nonexistent.lora")
			_ = cfg.SetAudioLoRAPath("nonexistent_audio.lora")
		})
	}
	wg.Wait()
}

func TestConcurrent_ClientLifecycle(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test skipped in -short mode")
	}
	libDir := getLocalLibDir()
	modelPath := os.Getenv("LITERTLM_TEST_MODEL")
	if modelPath == "" {
		modelPath = os.Getenv("LITERTLM_MODEL")
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
