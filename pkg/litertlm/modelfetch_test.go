package litertlm_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func TestFetchModel(t *testing.T) {
	data := []byte("top level fetch model test data")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(data)
	}))
	defer srv.Close()

	tmpDir := t.TempDir()
	ctx := context.Background()

	var progressCalled bool
	modelPath, err := litertlm.FetchModel(ctx, srv.URL+"/exported.litertlm",
		litertlm.WithModelDir(tmpDir),
		litertlm.WithModelFilename("custom-name.litertlm"),
		litertlm.WithModelProgress(func(downloaded, total int64, pct float64) {
			progressCalled = true
		}),
	)
	if err != nil {
		t.Fatalf("FetchModel: %v", err)
	}

	if filepath.Base(modelPath) != "custom-name.litertlm" {
		t.Errorf("modelPath base = %q, want custom-name.litertlm", filepath.Base(modelPath))
	}

	content, err := os.ReadFile(modelPath)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if string(content) != string(data) {
		t.Errorf("content = %q, want %q", string(content), string(data))
	}

	if !progressCalled {
		t.Error("expected progress callback to be called")
	}
}
