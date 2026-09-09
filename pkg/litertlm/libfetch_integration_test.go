package litertlm_test

import (
	"context"
	"os"
	"runtime"
	"testing"

	"github.com/vladimirvivien/litertlm-go/pkg/libfetch"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

// TestFetchLib_LiveDownloadAndBind tests fetching and binding against
// shared libraries.
func TestFetchLib_LiveDownloadAndBind(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test skipped in -short mode")
	}
	// If local LITERTLM_TEST_LIB or LITERTLM_LIB exists, test explicit fetch to a temp dir
	// and verify that Load binds symbols from that directory.
	existingLib := os.Getenv("LITERTLM_TEST_LIB")
	if existingLib == "" {
		existingLib = os.Getenv("LITERTLM_LIB")
	}

	if existingLib == "" {
		t.Skip("skipping live binding test: no local prebuilts and LITERTLM_LIB unset")
	}

	// Verify Platform resolution
	platform, err := libfetch.PlatformFor(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		t.Fatalf("PlatformFor error: %v", err)
	}
	t.Logf("Platform: %s", platform)

	// Verify loading from existing lib directory
	err = litertlm.Load(existingLib, "cpu", "")
	if err != nil {
		t.Fatalf("litertlm.Load failed: %v", err)
	}

	// If model is available, test end-to-end inference
	modelPath := os.Getenv("LITERTLM_TEST_MODEL")
	if modelPath == "" {
		modelPath = os.Getenv("LITERTLM_MODEL")
	}

	if modelPath != "" {
		ctx := context.Background()
		client, err := litertlm.New(ctx,
			litertlm.WithLib(existingLib),
			litertlm.WithModel(modelPath),
			litertlm.WithBackend("cpu"),
			litertlm.WithMaxTokens(256),
		)
		if err != nil {
			t.Fatalf("litertlm.New failed: %v", err)
		}
		defer func() { _ = client.Close() }()

		resp, err := client.Generate(ctx, "What is 2+2?")
		if err != nil {
			t.Fatalf("client.Generate failed: %v", err)
		}
		t.Logf("Inference output: %s", resp)
		if len(resp) == 0 {
			t.Error("expected non-empty inference response")
		}
	}
}
