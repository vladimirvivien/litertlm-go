package litertlm

import (
	"context"
	"testing"
)

func TestEmbeddingHandles_ZeroSafe(t *testing.T) {
	var settings EmbeddingEngineSettings
	settings.Delete()
	if err := settings.SetNumThreads(4); err == nil {
		t.Error("expected error on zero settings SetNumThreads")
	}
	if err := settings.SetAudioNumThreads(2); err == nil {
		t.Error("expected error on zero settings SetAudioNumThreads")
	}
	if err := settings.SetCacheDir("/tmp"); err == nil {
		t.Error("expected error on zero settings SetCacheDir")
	}
	if err := settings.SetLitertDispatchLibDir("/tmp"); err == nil {
		t.Error("expected error on zero settings SetLitertDispatchLibDir")
	}
	if err := settings.SetVisionLitertDispatchLibDir("/tmp"); err == nil {
		t.Error("expected error on zero settings SetVisionLitertDispatchLibDir")
	}
	if err := settings.SetAudioLitertDispatchLibDir("/tmp"); err == nil {
		t.Error("expected error on zero settings SetAudioLitertDispatchLibDir")
	}
	if err := settings.SetMaxInputLength(512); err == nil {
		t.Error("expected error on zero settings SetMaxInputLength")
	}
	if err := settings.SetMinInputLength(128); err == nil {
		t.Error("expected error on zero settings SetMinInputLength")
	}
	if err := settings.SetVisionTokensPerImage(280); err == nil {
		t.Error("expected error on zero settings SetVisionTokensPerImage")
	}
	if err := settings.SetActivationDataType(0); err == nil {
		t.Error("expected error on zero settings SetActivationDataType")
	}
	if _, err := NewEmbeddingEngineSettings("", "cpu", nil, nil); err == nil {
		t.Error("expected error on empty model path")
	}

	var opts EmbeddingOptions
	opts.Delete()
	if err := opts.SetNormalize(true); err == nil {
		t.Error("expected error on zero opts SetNormalize")
	}
	if _, err := opts.Normalize(); err == nil {
		t.Error("expected error on zero opts Normalize")
	}
	if err := opts.SetInsertSpecialTokens(true); err == nil {
		t.Error("expected error on zero opts SetInsertSpecialTokens")
	}
	if _, err := opts.InsertSpecialTokens(); err == nil {
		t.Error("expected error on zero opts InsertSpecialTokens")
	}
	if err := opts.SetOverflowStrategy(OverflowTruncate); err == nil {
		t.Error("expected error on zero opts SetOverflowStrategy")
	}
	if _, err := opts.OverflowStrategy(); err == nil {
		t.Error("expected error on zero opts OverflowStrategy")
	}
	if err := opts.SetOutputSize(128); err == nil {
		t.Error("expected error on zero opts SetOutputSize")
	}
	if _, err := opts.OutputSize(); err == nil {
		t.Error("expected error on zero opts OutputSize")
	}
	if err := opts.SetVisionTokensPerImage(64); err == nil {
		t.Error("expected error on zero opts SetVisionTokensPerImage")
	}
	if _, err := opts.VisionTokensPerImage(); err == nil {
		t.Error("expected error on zero opts VisionTokensPerImage")
	}

	var resp EmbeddingResponse
	resp.Delete()
	if _, err := resp.Size(); err == nil {
		t.Error("expected error on zero response Size")
	}
	if _, err := resp.Values(); err == nil {
		t.Error("expected error on zero response Values")
	}

	var resps EmbeddingResponses
	resps.Delete()
	if _, err := resps.Size(); err == nil {
		t.Error("expected error on zero responses Size")
	}
	if _, err := resps.At(0); err == nil {
		t.Error("expected error on zero responses At")
	}
	if _, err := resps.Values(); err == nil {
		t.Error("expected error on zero responses Values")
	}

	var engine EmbeddingEngine
	engine.Delete()
	if _, err := NewEmbeddingEngine(0); err == nil {
		t.Error("expected error on zero settings NewEmbeddingEngine")
	}
	if _, err := engine.ComputeEmbedding([]InputData{1}, 0); err == nil {
		t.Error("expected error on zero engine ComputeEmbedding")
	}
	if _, err := EmbeddingEngine(1).ComputeEmbedding(nil, 0); err == nil {
		t.Error("expected error on empty inputs ComputeEmbedding")
	}
	if _, err := engine.ComputeEmbeddingBatch([][]InputData{{1}}, 0); err == nil {
		t.Error("expected error on zero engine ComputeEmbeddingBatch")
	}
	if _, err := EmbeddingEngine(1).ComputeEmbeddingBatch(nil, 0); err == nil {
		t.Error("expected error on empty batch ComputeEmbeddingBatch")
	}
	if _, err := EmbeddingEngine(1).ComputeEmbeddingBatch([][]InputData{{}}, 0); err == nil {
		t.Error("expected error on empty row ComputeEmbeddingBatch")
	}
}

func TestEmbeddingClient_ValidationAndOptions(t *testing.T) {
	t.Setenv(envModel, "")
	if _, err := NewEmbeddingClient(context.Background()); err == nil {
		t.Error("expected error when model path is unset")
	}
	if _, err := NewEmbeddingClient(context.Background(), WithModelFD(3)); err == nil {
		t.Error("expected error when WithModelFD is passed to NewEmbeddingClient")
	}
	if _, err := NewEmbeddingClient(context.Background(), WithConfigFile("nonexistent_embed.json", "")); err == nil {
		t.Error("expected config error propagation in NewEmbeddingClient")
	}

	var nilClient *EmbeddingClient
	if err := nilClient.Close(); err != nil {
		t.Errorf("nil Close returned error: %v", err)
	}
	if nilClient.Engine() != 0 || nilClient.Settings() != 0 {
		t.Errorf("expected zero handles from nil client")
	}
	if _, err := nilClient.Capabilities(); err == nil {
		t.Error("expected error on nil Capabilities")
	}
	if _, err := nilClient.Embed(context.Background(), "hi"); err == nil {
		t.Error("expected error on nil Embed")
	}

	closedClient := &EmbeddingClient{closed: true}
	if err := closedClient.Close(); err != nil {
		t.Errorf("closed Close returned error: %v", err)
	}
	if _, err := closedClient.Embed(context.Background(), "hi"); err == nil {
		t.Error("expected error on closed Embed")
	}
	if _, err := closedClient.EmbedBatch(context.Background(), nil); err == nil {
		t.Error("expected error on empty EmbedBatch")
	}
	if _, err := closedClient.EmbedMultiBatch(context.Background(), nil); err == nil {
		t.Error("expected error on empty EmbedMultiBatch")
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := closedClient.Embed(canceled, "hi"); err == nil {
		t.Error("expected context cancellation error on Embed")
	}
	if _, err := closedClient.EmbedMultiBatch(canceled, [][]Part{{Text("hi")}}); err == nil {
		t.Error("expected context cancellation error on EmbedMultiBatch")
	}

	cfg := resolveEmbeddingConfig([]EmbeddingOption{
		WithNormalize(false),
		WithInsertSpecialTokens(false),
		WithOverflowStrategy(OverflowTruncate),
		WithOutputSize(128),
		WithEmbeddingVisionTokensPerImage(280),
	})
	if cfg.normalize == nil || *cfg.normalize {
		t.Errorf("normalize = %v, want false", cfg.normalize)
	}
	if cfg.insertSpecialTokens == nil || *cfg.insertSpecialTokens {
		t.Errorf("insertSpecialTokens = %v, want false", cfg.insertSpecialTokens)
	}
	if cfg.overflowStrategy == nil || *cfg.overflowStrategy != OverflowTruncate {
		t.Errorf("overflowStrategy = %v, want Truncate", cfg.overflowStrategy)
	}
	if cfg.outputSize == nil || *cfg.outputSize != 128 {
		t.Errorf("outputSize = %v, want 128", cfg.outputSize)
	}
	if cfg.visionTokensPerImage == nil || *cfg.visionTokensPerImage != 280 {
		t.Errorf("visionTokensPerImage = %v, want 280", cfg.visionTokensPerImage)
	}

	if _, err := partsToInputData(nil); err == nil {
		t.Error("expected error for empty parts")
	}
	if _, err := partsToInputData([]Part{{kind: partKind(99)}}); err == nil {
		t.Error("expected error for unknown part kind")
	}

	cCfg := clientConfig{}
	for _, opt := range []Option{
		WithVisionDispatchLibDir("/v"),
		WithAudioDispatchLibDir("/a"),
		WithMaxInputLength(256),
		WithMinInputLength(32),
	} {
		opt(&cCfg)
	}
	if cCfg.visionDispatchLibDir != "/v" || cCfg.audioDispatchLibDir != "/a" ||
		cCfg.maxInputLength == nil || *cCfg.maxInputLength != 256 ||
		cCfg.minInputLength == nil || *cCfg.minInputLength != 32 {
		t.Errorf("unexpected embedding clientConfig options: %+v", cCfg)
	}
}
