package litertlm_test

import (
	"context"
	"testing"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func TestModelCapabilities_SupportsModality(t *testing.T) {
	caps := &litertlm.ModelCapabilities{
		Path: "/path/to/model.litertlm",
		SupportedModalities: []litertlm.Modality{
			litertlm.ModalityText,
			litertlm.ModalityVision,
		},
	}

	if !caps.SupportsModality(litertlm.ModalityText) {
		t.Errorf("expected SupportsModality(ModalityText) to be true")
	}
	if !caps.SupportsModality(litertlm.ModalityVision) {
		t.Errorf("expected SupportsModality(ModalityVision) to be true")
	}
	if caps.SupportsModality(litertlm.ModalityAudio) {
		t.Errorf("expected SupportsModality(ModalityAudio) to be false")
	}
	if caps.SupportsModality(litertlm.ModalityVideo) {
		t.Errorf("expected SupportsModality(ModalityVideo) to be false")
	}
}

func TestModelCapabilities_Helpers(t *testing.T) {
	caps := &litertlm.ModelCapabilities{
		SupportedModalities: []litertlm.Modality{
			litertlm.ModalityText,
			litertlm.ModalityAudio,
		},
	}

	if !caps.SupportsText() {
		t.Errorf("SupportsText() = false, want true")
	}
	if caps.SupportsVision() {
		t.Errorf("SupportsVision() = true, want false")
	}
	if !caps.SupportsAudio() {
		t.Errorf("SupportsAudio() = false, want true")
	}
	if caps.SupportsVideo() {
		t.Errorf("SupportsVideo() = true, want false")
	}
}

func TestModelCapabilities_Nil(t *testing.T) {
	var caps *litertlm.ModelCapabilities
	if caps.SupportsModality(litertlm.ModalityText) {
		t.Errorf("nil caps SupportsModality() should be false")
	}
	if caps.SupportsText() {
		t.Errorf("nil caps SupportsText() should be false")
	}
	if caps.SupportsVision() {
		t.Errorf("nil caps SupportsVision() should be false")
	}
	if caps.SupportsAudio() {
		t.Errorf("nil caps SupportsAudio() should be false")
	}
	if caps.SupportsVideo() {
		t.Errorf("nil caps SupportsVideo() should be false")
	}
}

func TestLoadedFile_ZeroHandle(t *testing.T) {
	var f litertlm.LoadedFile

	// Should safely no-op without panicking
	f.Delete()

	if f.HasSpeculativeDecodingSupport() {
		t.Errorf("HasSpeculativeDecodingSupport on zero handle should be false")
	}
	if f.SupportsThinking() {
		t.Errorf("SupportsThinking on zero handle should be false")
	}
	if f.SupportsFunctionCalling() {
		t.Errorf("SupportsFunctionCalling on zero handle should be false")
	}
	if f.SamplerType() != litertlm.SamplerTypeUnspecified {
		t.Errorf("SamplerType on zero handle = %v, want SamplerTypeUnspecified", f.SamplerType())
	}
	if f.SamplerTemperature() != 0 {
		t.Errorf("SamplerTemperature on zero handle = %v, want 0", f.SamplerTemperature())
	}
	if f.SamplerTopK() != 0 {
		t.Errorf("SamplerTopK on zero handle = %v, want 0", f.SamplerTopK())
	}
	if f.SamplerTopP() != 0 {
		t.Errorf("SamplerTopP on zero handle = %v, want 0", f.SamplerTopP())
	}
	if f.SupportsInputModality(litertlm.ModalityText) {
		t.Errorf("SupportsInputModality on zero handle should be false")
	}
	if f.MaxVisionTokenBudget() != -1 {
		t.Errorf("MaxVisionTokenBudget on zero handle = %d, want -1", f.MaxVisionTokenBudget())
	}
	if mt, err := f.ModelType(); err == nil || mt != litertlm.ModelTypeUnknown {
		t.Errorf("ModelType zero = (%v, %v), want (ModelTypeUnknown, err)", mt, err)
	}
	if n, err := f.MaxContextTokens(); err == nil || n != 0 {
		t.Errorf("MaxContextTokens zero = (%d, %v), want (0, err)", n, err)
	}
	if f.IsDynamicContext() {
		t.Errorf("IsDynamicContext on zero handle should be false")
	}
	if lens, err := f.VisionSignatureLengths(); err == nil || lens != nil {
		t.Errorf("VisionSignatureLengths zero = (%v, %v), want error", lens, err)
	}
	if dim, err := f.EmbeddingDimension(); err == nil || dim != 0 {
		t.Errorf("EmbeddingDimension zero = (%d, %v), want (0, err)", dim, err)
	}
	if lens, err := f.EmbeddingSignatureLengths(); err == nil || lens != nil {
		t.Errorf("EmbeddingSignatureLengths zero = (%v, %v), want error", lens, err)
	}
	if b, err := f.ModalitySupportedBackends(litertlm.ModalityText); err == nil || b != nil {
		t.Errorf("ModalitySupportedBackends zero = (%v, %v), want error", b, err)
	}
	if nb, err := f.ModalityNPUBrand(litertlm.ModalityText); err == nil || nb != litertlm.NPUBrandUnknown {
		t.Errorf("ModalityNPUBrand zero = (%v, %v), want (NPUBrandUnknown, err)", nb, err)
	}
	if soc, err := f.ModalitySoCName(litertlm.ModalityText); err == nil || soc != "" {
		t.Errorf("ModalitySoCName zero = (%q, %v), want error", soc, err)
	}
	if ver, err := f.MinRuntimeVersion(); err == nil || ver != "" {
		t.Errorf("MinRuntimeVersion zero = (%q, %v), want error", ver, err)
	}
	if f.Snapshot("dummy") != nil {
		t.Errorf("Snapshot on zero handle should be nil")
	}
}

func TestOpenLoadedFile_EmptyPath(t *testing.T) {
	_, err := litertlm.OpenLoadedFile("")
	if err == nil {
		t.Fatal("expected error for empty model path, got nil")
	}
}

func TestInspectModel_Integration(t *testing.T) {
	libDir, modelPath := requireTestModel(t)

	if err := litertlm.Load(libDir, "cpu", ""); err != nil {
		t.Fatalf("Load: %v", err)
	}

	caps, err := litertlm.InspectModel(modelPath)
	if err != nil {
		t.Fatalf("InspectModel(%q): %v", modelPath, err)
	}

	if caps == nil {
		t.Fatal("expected non-nil ModelCapabilities")
	}
	if caps.Path != modelPath {
		t.Errorf("caps.Path = %q, want %q", caps.Path, modelPath)
	}
	if !caps.SupportsText() {
		t.Errorf("expected test model to support text modality")
	}

	// Also verify Client.Capabilities() method
	client, err := litertlm.New(context.Background(),
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
	)
	if err != nil {
		t.Fatalf("New client: %v", err)
	}
	defer func() { _ = client.Close() }()

	clientCaps, err := client.Capabilities()
	if err != nil {
		t.Fatalf("client.Capabilities(): %v", err)
	}
	if clientCaps.Path != modelPath {
		t.Errorf("clientCaps.Path = %q, want %q", clientCaps.Path, modelPath)
	}
}
