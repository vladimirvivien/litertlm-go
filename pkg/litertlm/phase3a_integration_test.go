package litertlm_test

import (
	"context"
	"errors"
	"math"
	"os"
	"testing"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func TestLive_Phase3a_ModelInfoFull(t *testing.T) {
	libDir, modelPath := requireTestModel(t)
	if err := litertlm.Load(libDir, "cpu", ""); err != nil {
		t.Fatalf("Load: %v", err)
	}
	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	f, err := litertlm.OpenLoadedFile(modelPath)
	if err != nil {
		t.Fatalf("OpenLoadedFile(%q): %v", modelPath, err)
	}
	defer f.Delete()

	mt, err := f.ModelType()
	if err != nil {
		t.Fatalf("ModelType: %v", err)
	}
	if mt != litertlm.ModelTypeLLM && mt != litertlm.ModelTypeEmbedding {
		t.Errorf("unexpected ModelType = %v", mt)
	}

	maxCtx, ctxErr := f.MaxContextTokens()
	if ctxErr != nil && !errors.Is(ctxErr, litertlm.ErrNotFound) {
		t.Fatalf("MaxContextTokens unexpected error: %v", ctxErr)
	}
	if ctxErr == nil && maxCtx <= 0 {
		t.Errorf("MaxContextTokens = %d, want > 0", maxCtx)
	}

	dynamic := f.IsDynamicContext()
	minVer, err := f.MinRuntimeVersion()
	if err != nil {
		t.Fatalf("MinRuntimeVersion: %v", err)
	}

	textBackends, err := f.ModalitySupportedBackends(litertlm.ModalityText)
	if err != nil {
		t.Fatalf("ModalitySupportedBackends(Text): %v", err)
	}
	if len(textBackends) == 0 {
		t.Errorf("expected at least one backend for Text modality")
	}

	brand, err := f.ModalityNPUBrand(litertlm.ModalityText)
	if err != nil {
		t.Fatalf("ModalityNPUBrand(Text): %v", err)
	}
	soc, err := f.ModalitySoCName(litertlm.ModalityText)
	if err != nil {
		t.Fatalf("ModalitySoCName(Text): %v", err)
	}

	visLens, visErr := f.VisionSignatureLengths()
	if visErr != nil && !errors.Is(visErr, litertlm.ErrNotFound) {
		t.Fatalf("VisionSignatureLengths unexpected error: %v", visErr)
	}

	if mt == litertlm.ModelTypeLLM {
		if _, dimErr := f.EmbeddingDimension(); !errors.Is(dimErr, litertlm.ErrNotFound) {
			t.Errorf("LLM EmbeddingDimension err = %v, want ErrNotFound", dimErr)
		}
		if _, embErr := f.EmbeddingSignatureLengths(); !errors.Is(embErr, litertlm.ErrNotFound) {
			t.Errorf("LLM EmbeddingSignatureLengths err = %v, want ErrNotFound", embErr)
		}
	}

	caps, err := litertlm.InspectModel(modelPath)
	if err != nil {
		t.Fatalf("InspectModel: %v", err)
	}
	if caps.ModelType != mt {
		t.Errorf("caps.ModelType = %v, want %v", caps.ModelType, mt)
	}
	if caps.IsDynamicContext != dynamic {
		t.Errorf("caps.IsDynamicContext = %v, want %v", caps.IsDynamicContext, dynamic)
	}
	if caps.MinRuntimeVersion != minVer {
		t.Errorf("caps.MinRuntimeVersion = %q, want %q", caps.MinRuntimeVersion, minVer)
	}
	if ctxErr == nil && caps.MaxContextTokens != maxCtx {
		t.Errorf("caps.MaxContextTokens = %d, want %d", caps.MaxContextTokens, maxCtx)
	}
	if len(caps.ModalityBackends[litertlm.ModalityText]) != len(textBackends) {
		t.Errorf("caps.ModalityBackends[Text] = %v, want %v", caps.ModalityBackends[litertlm.ModalityText], textBackends)
	}
	t.Logf("model=%s type=%s minVer=%q maxCtx=%d dynamic=%v backends=%v brand=%s soc=%q visionLens=%v",
		modelPath, mt, minVer, caps.MaxContextTokens, dynamic, textBackends, brand, soc, visLens)
}

func TestLive_Phase3a_EmbeddingOptionsAndSettings(t *testing.T) {
	libDir, modelPath := requireTestModel(t)
	if err := litertlm.Load(libDir, "cpu", ""); err != nil {
		t.Fatalf("Load: %v", err)
	}
	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	opts, err := litertlm.NewEmbeddingOptions()
	if err != nil {
		t.Fatalf("NewEmbeddingOptions: %v", err)
	}
	defer opts.Delete()

	if norm, normErr := opts.Normalize(); normErr != nil || !norm {
		t.Errorf("default Normalize = (%v, %v), want (true, nil)", norm, normErr)
	}
	if special, spErr := opts.InsertSpecialTokens(); spErr != nil || !special {
		t.Errorf("default InsertSpecialTokens = (%v, %v), want (true, nil)", special, spErr)
	}
	if strategy, stErr := opts.OverflowStrategy(); stErr != nil || strategy != litertlm.OverflowError {
		t.Errorf("default OverflowStrategy = (%v, %v), want (Error, nil)", strategy, stErr)
	}
	if _, outErr := opts.OutputSize(); !errors.Is(outErr, litertlm.ErrNotFound) {
		t.Errorf("default OutputSize err = %v, want ErrNotFound", outErr)
	}
	if _, visErr := opts.VisionTokensPerImage(); !errors.Is(visErr, litertlm.ErrNotFound) {
		t.Errorf("default VisionTokensPerImage err = %v, want ErrNotFound", visErr)
	}

	if err = opts.SetNormalize(false); err != nil {
		t.Fatalf("SetNormalize(false): %v", err)
	}
	if norm, normErr := opts.Normalize(); normErr != nil || norm {
		t.Errorf("after SetNormalize(false) = (%v, %v)", norm, normErr)
	}
	if err = opts.SetInsertSpecialTokens(false); err != nil {
		t.Fatalf("SetInsertSpecialTokens(false): %v", err)
	}
	if special, spErr := opts.InsertSpecialTokens(); spErr != nil || special {
		t.Errorf("after SetInsertSpecialTokens(false) = (%v, %v)", special, spErr)
	}
	for _, s := range []litertlm.OverflowStrategy{
		litertlm.OverflowTruncate,
		litertlm.OverflowChunkAndAverage,
	} {
		if err = opts.SetOverflowStrategy(s); err != nil {
			t.Fatalf("SetOverflowStrategy(%v): %v", s, err)
		}
		if got, gotErr := opts.OverflowStrategy(); gotErr != nil || got != s {
			t.Errorf("OverflowStrategy = (%v, %v), want (%v, nil)", got, gotErr, s)
		}
	}

	if err = opts.SetOutputSize(128); err != nil {
		t.Fatalf("SetOutputSize(128): %v", err)
	}
	if got, gotErr := opts.OutputSize(); gotErr != nil || got != 128 {
		t.Errorf("OutputSize = (%d, %v), want (128, nil)", got, gotErr)
	}
	if err = opts.SetOutputSize(0); err != nil {
		t.Fatalf("SetOutputSize(0): %v", err)
	}
	if _, gotErr := opts.OutputSize(); !errors.Is(gotErr, litertlm.ErrNotFound) {
		t.Errorf("OutputSize after unset err = %v, want ErrNotFound", gotErr)
	}

	if err = opts.SetVisionTokensPerImage(70); err != nil {
		t.Fatalf("SetVisionTokensPerImage(70): %v", err)
	}
	if got, gotErr := opts.VisionTokensPerImage(); gotErr != nil || got != 70 {
		t.Errorf("VisionTokensPerImage = (%d, %v), want (70, nil)", got, gotErr)
	}
	if err = opts.SetVisionTokensPerImage(-1); err != nil {
		t.Fatalf("SetVisionTokensPerImage(-1): %v", err)
	}
	if _, gotErr := opts.VisionTokensPerImage(); !errors.Is(gotErr, litertlm.ErrNotFound) {
		t.Errorf("VisionTokensPerImage after unset err = %v, want ErrNotFound", gotErr)
	}

	settingsModel := modelPath
	if emb := os.Getenv("LITERTLM_TEST_EMBED_MODEL"); emb != "" {
		settingsModel = emb
	}
	settings, sErr := litertlm.NewEmbeddingEngineSettings(settingsModel, "cpu", nil, nil)
	if sErr != nil {
		if _, ok := errors.AsType[*litertlm.CError](sErr); !ok {
			t.Fatalf("expected *CError when creating embedding settings on %q, got %T: %v", settingsModel, sErr, sErr)
		}
		return
	}
	defer settings.Delete()

	for name, callErr := range map[string]error{
		"SetNumThreads":                settings.SetNumThreads(2),
		"SetAudioNumThreads":           settings.SetAudioNumThreads(2),
		"SetCacheDir":                  settings.SetCacheDir(t.TempDir()),
		"SetLitertDispatchLibDir":      settings.SetLitertDispatchLibDir(t.TempDir()),
		"SetVisionLitertDispatchDir":   settings.SetVisionLitertDispatchLibDir(t.TempDir()),
		"SetAudioLitertDispatchLibDir": settings.SetAudioLitertDispatchLibDir(t.TempDir()),
		"SetMaxInputLength":            settings.SetMaxInputLength(512),
		"SetMinInputLength":            settings.SetMinInputLength(64),
		"SetVisionTokensPerImage":      settings.SetVisionTokensPerImage(280),
		"SetActivationDataType":        settings.SetActivationDataType(0),
	} {
		if callErr != nil {
			t.Errorf("EmbeddingEngineSettings.%s: %v", name, callErr)
		}
	}

	if os.Getenv("LITERTLM_TEST_EMBED_MODEL") == "" {
		if eng, engErr := litertlm.NewEmbeddingEngine(settings); engErr == nil {
			eng.Delete()
		} else {
			if _, ok := errors.AsType[*litertlm.CError](engErr); !ok {
				t.Errorf("expected *CError opening LLM model with EmbeddingEngine, got %T: %v", engErr, engErr)
			}
		}
	}
}

func TestLive_Phase3a_EmbeddingClientEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test skipped in -short mode")
	}
	embedModel := os.Getenv("LITERTLM_TEST_EMBED_MODEL")
	if embedModel == "" {
		t.Skip("LITERTLM_TEST_EMBED_MODEL not set; skipping end-to-end embedding inference")
	}
	libDir := os.Getenv("LITERTLM_TEST_LIB")
	if libDir == "" {
		libDir = os.Getenv("LITERTLM_LIB")
	}

	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	ctx := context.Background()
	client, err := litertlm.NewEmbeddingClient(ctx,
		litertlm.WithLib(libDir),
		litertlm.WithModel(embedModel),
		litertlm.WithBackend("cpu"),
	)
	if err != nil {
		t.Fatalf("NewEmbeddingClient: %v", err)
	}
	defer func() { _ = client.Close() }()

	vec, err := client.Embed(ctx, "Go concurrency with channels",
		litertlm.WithNormalize(true),
		litertlm.WithOverflowStrategy(litertlm.OverflowChunkAndAverage),
	)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vec) == 0 {
		t.Fatal("expected non-empty embedding vector")
	}
	var sumSq float64
	for _, x := range vec {
		sumSq += float64(x) * float64(x)
	}
	if math.Abs(sumSq-1.0) > 1e-3 {
		t.Errorf("L2 norm^2 = %f, want ~1.0", sumSq)
	}

	trunc, err := client.EmbedBatch(ctx, []string{"hello world", "goodbye world"},
		litertlm.WithNormalize(true),
		litertlm.WithOutputSize(64),
	)
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	if len(trunc) != 2 || len(trunc[0]) != 64 || len(trunc[1]) != 64 {
		t.Errorf("EmbedBatch truncated shapes = %d (%d, %d)", len(trunc), len(trunc[0]), len(trunc[1]))
	}
}
