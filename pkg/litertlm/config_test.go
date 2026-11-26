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

func TestWithConfigFile_Phase2Fields(t *testing.T) {
	tmp := t.TempDir()
	configPath := filepath.Join(tmp, "config.json")

	jsonContent := `{
		"default": {
			"speculative_decoding_mode": "adaptive",
			"speculative_decoding_lenience": 0.25,
			"enable_ynnpack": true,
			"max_vision_tokens_per_image": 512,
			"metal_residency_set": true,
			"gpu_decode_steps_per_sync": 4,
			"gpu_wait_for_weight_uploads": false,
			"use_ringbuffers_local_attention": true,
			"single_threaded_execution": true,
			"selected_signatures": ["prefill_1024", "decode"]
		}
	}`
	if err := os.WriteFile(configPath, []byte(jsonContent), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	cfg := clientConfig{}
	WithConfigFile(configPath, "")(&cfg)
	if cfg.err != nil {
		t.Fatalf("unexpected error: %v", cfg.err)
	}
	if cfg.speculativeDecoding == nil || cfg.speculativeDecoding.Mode != SpeculativeDecodingAdaptive || cfg.speculativeDecoding.Lenience != 0.25 {
		t.Fatalf("speculativeDecoding = %+v", cfg.speculativeDecoding)
	}
	if cfg.enableYNNPACK == nil || !*cfg.enableYNNPACK {
		t.Errorf("enableYNNPACK = %v, want true", cfg.enableYNNPACK)
	}
	if cfg.maxVisionTokensPerImage == nil || *cfg.maxVisionTokensPerImage != 512 {
		t.Errorf("maxVisionTokensPerImage = %v, want 512", cfg.maxVisionTokensPerImage)
	}
	if cfg.metalResidencySet == nil || !*cfg.metalResidencySet {
		t.Errorf("metalResidencySet = %v, want true", cfg.metalResidencySet)
	}
	if cfg.gpuDecodeStepsPerSync == nil || *cfg.gpuDecodeStepsPerSync != 4 {
		t.Errorf("gpuDecodeStepsPerSync = %v, want 4", cfg.gpuDecodeStepsPerSync)
	}
	if cfg.gpuWaitForWeightUploads == nil || *cfg.gpuWaitForWeightUploads {
		t.Errorf("gpuWaitForWeightUploads = %v, want false", cfg.gpuWaitForWeightUploads)
	}
	if cfg.useRingbuffersLocalAttn == nil || !*cfg.useRingbuffersLocalAttn {
		t.Errorf("useRingbuffersLocalAttn = %v, want true", cfg.useRingbuffersLocalAttn)
	}
	if cfg.singleThreadedExecution == nil || !*cfg.singleThreadedExecution {
		t.Errorf("singleThreadedExecution = %v, want true", cfg.singleThreadedExecution)
	}
	if len(cfg.selectedSignatures) != 2 || cfg.selectedSignatures[0] != "prefill_1024" || cfg.selectedSignatures[1] != "decode" {
		t.Errorf("selectedSignatures = %v", cfg.selectedSignatures)
	}

	badPath := filepath.Join(tmp, "bad_mode.json")
	if err := os.WriteFile(badPath, []byte(`{"default":{"speculative_decoding_mode":"invalid"}}`), 0o600); err != nil {
		t.Fatalf("WriteFile bad: %v", err)
	}
	badCfg := clientConfig{}
	WithConfigFile(badPath, "")(&badCfg)
	if badCfg.err == nil {
		t.Fatal("expected error for invalid speculative_decoding_mode")
	}
}

func TestWithConfigFile_EmbeddingFields(t *testing.T) {
	tmp := t.TempDir()
	configPath := filepath.Join(tmp, "embed_config.json")

	jsonContent := `{
		"default": {
			"vision_dispatch_lib_dir": "/opt/npu/vision",
			"audio_dispatch_lib_dir": "/opt/npu/audio",
			"max_input_length": 512,
			"min_input_length": 64
		}
	}`
	if err := os.WriteFile(configPath, []byte(jsonContent), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	cfg := clientConfig{}
	WithConfigFile(configPath, "")(&cfg)
	if cfg.err != nil {
		t.Fatalf("unexpected error: %v", cfg.err)
	}
	if cfg.visionDispatchLibDir != "/opt/npu/vision" {
		t.Errorf("visionDispatchLibDir = %q", cfg.visionDispatchLibDir)
	}
	if cfg.audioDispatchLibDir != "/opt/npu/audio" {
		t.Errorf("audioDispatchLibDir = %q", cfg.audioDispatchLibDir)
	}
	if cfg.maxInputLength == nil || *cfg.maxInputLength != 512 {
		t.Errorf("maxInputLength = %v, want 512", cfg.maxInputLength)
	}
	if cfg.minInputLength == nil || *cfg.minInputLength != 64 {
		t.Errorf("minInputLength = %v, want 64", cfg.minInputLength)
	}
}
