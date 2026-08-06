package litertlm

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWithConfigFile_ParsingAndOverrides(t *testing.T) {
	tmp := t.TempDir()
	configPath := filepath.Join(tmp, "config.json")

	jsonContent := `{
		"default": {
			"backend": "cpu",
			"max_tokens": 2048,
			"num_threads": 4,
			"sampler_type": "top_p",
			"temperature": 0.7,
			"top_p": 0.9
		},
		"models": {
			"gemma": {
				"backend": "cpu",
				"max_tokens": 1024,
				"num_threads": 8,
				"sampler_type": "greedy",
				"temperature": 0.0
			},
			"qwen": {
				"backend": "gpu",
				"max_tokens": 4096,
				"sampler_type": "top_k",
				"top_k": 20
			}
		}
	}`

	if err := os.WriteFile(configPath, []byte(jsonContent), 0644); err != nil {
		t.Fatalf("failed to write test config file: %v", err)
	}

	t.Run("default configuration without model match", func(t *testing.T) {
		cfg := clientConfig{}
		WithConfigFile(configPath, "")(&cfg)
		if cfg.err != nil {
			t.Fatalf("unexpected error: %v", cfg.err)
		}
		if cfg.backend != "cpu" {
			t.Errorf("backend = %q, want cpu", cfg.backend)
		}
		if cfg.maxTokens != 2048 {
			t.Errorf("maxTokens = %d, want 2048", cfg.maxTokens)
		}
		if cfg.numThreads == nil || *cfg.numThreads != 4 {
			t.Errorf("numThreads = %v, want 4", cfg.numThreads)
		}
		if cfg.defaultSampler == nil {
			t.Fatal("expected defaultSampler to be populated")
		}
		if cfg.defaultSampler.Type != SamplerTopP {
			t.Errorf("sampler.Type = %v, want SamplerTopP", cfg.defaultSampler.Type)
		}
		if cfg.defaultSampler.Temperature != 0.7 {
			t.Errorf("sampler.Temperature = %f, want 0.7", cfg.defaultSampler.Temperature)
		}
		if cfg.defaultSampler.TopP != 0.9 {
			t.Errorf("sampler.TopP = %f, want 0.9", cfg.defaultSampler.TopP)
		}
	})

	t.Run("override with gemma model match", func(t *testing.T) {
		cfg := clientConfig{}
		WithConfigFile(configPath, "gemma")(&cfg)
		if cfg.err != nil {
			t.Fatalf("unexpected error: %v", cfg.err)
		}
		if cfg.backend != "cpu" {
			t.Errorf("backend = %q, want cpu", cfg.backend)
		}
		if cfg.maxTokens != 1024 {
			t.Errorf("maxTokens = %d, want 1024", cfg.maxTokens)
		}
		if cfg.numThreads == nil || *cfg.numThreads != 8 {
			t.Errorf("numThreads = %v, want 8", cfg.numThreads)
		}
		if cfg.defaultSampler == nil {
			t.Fatal("expected defaultSampler to be populated")
		}
		if cfg.defaultSampler.Type != SamplerGreedy {
			t.Errorf("sampler.Type = %v, want SamplerGreedy", cfg.defaultSampler.Type)
		}
		if cfg.defaultSampler.Temperature != 0.0 {
			t.Errorf("sampler.Temperature = %f, want 0.0", cfg.defaultSampler.Temperature)
		}
	})

	t.Run("override with qwen model match", func(t *testing.T) {
		cfg := clientConfig{}
		WithConfigFile(configPath, "qwen")(&cfg)
		if cfg.err != nil {
			t.Fatalf("unexpected error: %v", cfg.err)
		}
		if cfg.backend != "gpu" {
			t.Errorf("backend = %q, want gpu", cfg.backend)
		}
		if cfg.maxTokens != 4096 {
			t.Errorf("maxTokens = %d, want 4096", cfg.maxTokens)
		}
		if cfg.numThreads == nil || *cfg.numThreads != 4 { // inherited from default
			t.Errorf("numThreads = %v, want 4", cfg.numThreads)
		}
		if cfg.defaultSampler == nil {
			t.Fatal("expected defaultSampler to be populated")
		}
		if cfg.defaultSampler.Type != SamplerTopK {
			t.Errorf("sampler.Type = %v, want SamplerTopK", cfg.defaultSampler.Type)
		}
		if cfg.defaultSampler.TopK != 20 {
			t.Errorf("sampler.TopK = %d, want 20", cfg.defaultSampler.TopK)
		}
	})
}

func TestWithConfigFile_ErrorPropagation(t *testing.T) {
	t.Run("nonexistent file", func(t *testing.T) {
		cfg := clientConfig{}
		WithConfigFile("nonexistent_config_file.json", "")(&cfg)
		if cfg.err == nil {
			t.Fatal("expected error for nonexistent file but got nil")
		}
	})

	t.Run("invalid json file", func(t *testing.T) {
		tmp := t.TempDir()
		configPath := filepath.Join(tmp, "invalid.json")
		if err := os.WriteFile(configPath, []byte("{invalid-json"), 0644); err != nil {
			t.Fatalf("failed to write test config file: %v", err)
		}

		cfg := clientConfig{}
		WithConfigFile(configPath, "")(&cfg)
		if cfg.err == nil {
			t.Fatal("expected error for invalid json but got nil")
		}
	})

	t.Run("propagates to client.New constructor", func(t *testing.T) {
		ctx := context.Background()
		_, err := New(ctx, WithModel("dummy.litertlm"), WithConfigFile("nonexistent_config_file.json", ""))
		if err == nil {
			t.Fatal("expected error during client creation but got nil")
		}
		expect := "configuration error"
		if !strings.Contains(err.Error(), expect) {
			t.Errorf("expected error to contain %q, got %q", expect, err.Error())
		}
	})
}
