package modelfetch

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestResolveModelIdentifier(t *testing.T) {
	tests := []struct {
		in           string
		wantURL      string
		wantFilename string
		wantErr      bool
	}{
		{
			in:           "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it-gpu.litertlm",
			wantURL:      "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it-gpu.litertlm",
			wantFilename: "gemma-4-E4B-it-gpu.litertlm",
		},
		{
			in:           "hf:google/gemma-3-1b-it:model.litertlm",
			wantURL:      "https://huggingface.co/google/gemma-3-1b-it/resolve/main/model.litertlm",
			wantFilename: "model.litertlm",
		},
		{
			in:           "litert-community/gemma-4-E4B-it-litert-lm/gemma-4-E4B-it-gpu.litertlm",
			wantURL:      "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it-gpu.litertlm",
			wantFilename: "gemma-4-E4B-it-gpu.litertlm",
		},
		{
			in:           "litert-community/gemma-4-E4B-it-litert-lm:gpu",
			wantURL:      "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it-gpu.litertlm",
			wantFilename: "gemma-4-E4B-it-gpu.litertlm",
		},
		{
			in:           "litert-community/gemma-4-E4B-it-litert-lm:cpu",
			wantURL:      "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm",
			wantFilename: "gemma-4-E4B-it.litertlm",
		},
		{
			in:           "litert-community/gemma-4-E4B-it-litert-lm",
			wantURL:      "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm",
			wantFilename: "gemma-4-E4B-it.litertlm",
		},
		{
			in:           "litert-community/gemma-4-E2B-it-litert-lm",
			wantURL:      "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm",
			wantFilename: "gemma-4-E2B-it.litertlm",
		},
		{
			in:           "litert-community/gemma-4-12B-it-litert-lm",
			wantURL:      "https://huggingface.co/litert-community/gemma-4-12B-it-litert-lm/resolve/main/gemma-4-12B-it.litertlm",
			wantFilename: "gemma-4-12B-it.litertlm",
		},
		{
			in:           "litert-community/gemma-4-31B-it-litert-lm",
			wantURL:      "https://huggingface.co/litert-community/gemma-4-31B-it-litert-lm/resolve/main/gemma-4-31B-it.litertlm",
			wantFilename: "gemma-4-31B-it.litertlm",
		},
		{
			in:           "litert-community/gemma3-1b-it-int4",
			wantURL:      "https://huggingface.co/litert-community/gemma3-1b-it-int4/resolve/main/gemma3-1b-it-int4.litertlm",
			wantFilename: "gemma3-1b-it-int4.litertlm",
		},
		{
			in:           "gemma-4-E4B-it",
			wantURL:      "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm",
			wantFilename: "gemma-4-E4B-it.litertlm",
		},
		{
			in:           "gemma-4-E4B-it-gpu",
			wantURL:      "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it-gpu.litertlm",
			wantFilename: "gemma-4-E4B-it-gpu.litertlm",
		},
		{
			in:           "gemma-4-E2B-it",
			wantURL:      "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm",
			wantFilename: "gemma-4-E2B-it.litertlm",
		},
		{
			in:           "gemma-4-12B-it",
			wantURL:      "https://huggingface.co/litert-community/gemma-4-12B-it-litert-lm/resolve/main/gemma-4-12B-it.litertlm",
			wantFilename: "gemma-4-12B-it.litertlm",
		},
		{
			in:           "gemma-4-31B-it",
			wantURL:      "https://huggingface.co/litert-community/gemma-4-31B-it-litert-lm/resolve/main/gemma-4-31B-it.litertlm",
			wantFilename: "gemma-4-31B-it.litertlm",
		},
		{
			in:      "",
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.in, func(t *testing.T) {
			got, err := ResolveModelIdentifier(tc.in)
			if (err != nil) != tc.wantErr {
				t.Fatalf("ResolveModelIdentifier(%q) err = %v, wantErr = %v", tc.in, err, tc.wantErr)
			}
			if !tc.wantErr {
				if got.URL != tc.wantURL {
					t.Errorf("URL = %q, want %q", got.URL, tc.wantURL)
				}
				if got.Filename != tc.wantFilename {
					t.Errorf("Filename = %q, want %q", got.Filename, tc.wantFilename)
				}
			}
		})
	}
}

func TestResolveModelIdentifierWithVariant(t *testing.T) {
	tests := []struct {
		name         string
		in           string
		variant      string
		wantURL      string
		wantFilename string
	}{
		{
			name:         "gemma 4 4b with GPU variant option",
			in:           "litert-community/gemma-4-E4B-it-litert-lm",
			variant:      "gpu",
			wantURL:      "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it-gpu.litertlm",
			wantFilename: "gemma-4-E4B-it-gpu.litertlm",
		},
		{
			name:         "gemma 4 4b with CPU variant option",
			in:           "litert-community/gemma-4-E4B-it-litert-lm",
			variant:      "cpu",
			wantURL:      "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm",
			wantFilename: "gemma-4-E4B-it.litertlm",
		},
		{
			name:         "gemma 4 2b shorthand with GPU variant option",
			in:           "gemma-4-E2B-it",
			variant:      "gpu",
			wantURL:      "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-gpu.litertlm",
			wantFilename: "gemma-4-E2B-it-gpu.litertlm",
		},
		{
			name:         "gemma 4 12b with web variant option",
			in:           "litert-community/gemma-4-12B-it-litert-lm",
			variant:      "web",
			wantURL:      "https://huggingface.co/litert-community/gemma-4-12B-it-litert-lm/resolve/main/gemma-4-12B-it-web.litertlm",
			wantFilename: "gemma-4-12B-it-web.litertlm",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ResolveModelIdentifierWithVariant(tc.in, tc.variant)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got.URL != tc.wantURL {
				t.Errorf("URL = %q, want %q", got.URL, tc.wantURL)
			}
			if got.Filename != tc.wantFilename {
				t.Errorf("Filename = %q, want %q", got.Filename, tc.wantFilename)
			}
		})
	}
}

func TestFetch_DirectDownload(t *testing.T) {
	testData := []byte("dummy litertlm model binary data for test")
	var progressCalled int32

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", strconv.Itoa(len(testData)))
		_, _ = w.Write(testData)
	}))
	defer srv.Close()

	tmpDir := t.TempDir()
	ctx := context.Background()

	filePath, err := Fetch(ctx, srv.URL+"/test-model.litertlm",
		WithDir(tmpDir),
		WithProgress(func(downloaded, total int64, pct float64) {
			atomic.AddInt32(&progressCalled, 1)
		}),
	)
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}

	if filepath.Base(filePath) != "test-model.litertlm" {
		t.Errorf("unexpected filepath: %s", filePath)
	}

	content, err := os.ReadFile(filePath)
	if err != nil {
		t.Fatalf("read destination: %v", err)
	}
	if string(content) != string(testData) {
		t.Errorf("content mismatch: got %q, want %q", string(content), string(testData))
	}

	if atomic.LoadInt32(&progressCalled) == 0 {
		t.Error("expected progress callback to be called at least once")
	}
}

func TestFetch_WithGPUAndCPUOptions(t *testing.T) {
	testData := []byte("model binary data")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", strconv.Itoa(len(testData)))
		_, _ = w.Write(testData)
	}))
	defer srv.Close()

	tmpDir := t.TempDir()
	ctx := context.Background()

	// Test WithGPU option
	filePathGPU, err := Fetch(ctx, srv.URL+"/gemma-4-E4B-it.litertlm",
		WithDir(tmpDir),
		WithGPU(),
	)
	if err != nil {
		t.Fatalf("Fetch WithGPU: %v", err)
	}
	if filepath.Base(filePathGPU) != "gemma-4-E4B-it-gpu.litertlm" {
		t.Errorf("expected GPU filename, got %s", filepath.Base(filePathGPU))
	}
}

func TestFetch_Resume(t *testing.T) {
	fullData := []byte("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rangeHeader := r.Header.Get("Range")
		if rangeHeader != "" && strings.HasPrefix(rangeHeader, "bytes=") {
			rangeSpec := strings.TrimPrefix(rangeHeader, "bytes=")
			parts := strings.Split(rangeSpec, "-")
			start, _ := strconv.Atoi(parts[0])
			if start >= len(fullData) {
				w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
				return
			}
			chunk := fullData[start:]
			w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, len(fullData)-1, len(fullData)))
			w.Header().Set("Content-Length", strconv.Itoa(len(chunk)))
			w.WriteHeader(http.StatusPartialContent)
			_, _ = w.Write(chunk)
			return
		}

		w.Header().Set("Content-Length", strconv.Itoa(len(fullData)))
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(fullData)
	}))
	defer srv.Close()

	tmpDir := t.TempDir()
	ctx := context.Background()

	// Pre-create partial download file
	partPath := filepath.Join(tmpDir, "resumed-model.litertlm.download")
	if err := os.WriteFile(partPath, fullData[:10], 0o644); err != nil {
		t.Fatalf("write partial: %v", err)
	}

	filePath, err := Fetch(ctx, srv.URL+"/resumed-model.litertlm",
		WithDir(tmpDir),
	)
	if err != nil {
		t.Fatalf("Fetch resume: %v", err)
	}

	content, err := os.ReadFile(filePath)
	if err != nil {
		t.Fatalf("read destination: %v", err)
	}
	if string(content) != string(fullData) {
		t.Errorf("content mismatch after resume: got %q, want %q", string(content), string(fullData))
	}
}

func TestFetch_SHA256Checksum(t *testing.T) {
	data := []byte("checksum verification test data")
	sum := sha256.Sum256(data)
	correctHex := hex.EncodeToString(sum[:])
	wrongHex := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(data)
	}))
	defer srv.Close()

	tmpDir := t.TempDir()
	ctx := context.Background()

	// 1. Matching SHA-256
	filePath, err := Fetch(ctx, srv.URL+"/checksum.litertlm",
		WithDir(tmpDir),
		WithSHA256(correctHex),
	)
	if err != nil {
		t.Fatalf("Fetch with correct SHA256 failed: %v", err)
	}
	if _, statErr := os.Stat(filePath); statErr != nil {
		t.Errorf("expected destination file to exist: %v", statErr)
	}

	// 2. Mismatched SHA-256 should error and remove target file
	tmpDir2 := t.TempDir()
	_, err = Fetch(ctx, srv.URL+"/badchecksum.litertlm",
		WithDir(tmpDir2),
		WithSHA256(wrongHex),
	)
	if err == nil {
		t.Fatal("expected error on SHA-256 mismatch but got nil")
	}
	if !strings.Contains(err.Error(), "checksum mismatch") {
		t.Errorf("expected checksum mismatch error, got: %v", err)
	}
}

func TestFetch_AuthHeader(t *testing.T) {
	var authReceived string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authReceived = r.Header.Get("Authorization")
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()

	tmpDir := t.TempDir()
	ctx := context.Background()

	_, err := Fetch(ctx, srv.URL+"/authed.litertlm",
		WithDir(tmpDir),
		WithAuthToken("hf_test_secret_token_123"),
	)
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}

	if authReceived != "Bearer hf_test_secret_token_123" {
		t.Errorf("Authorization header = %q, want %q", authReceived, "Bearer hf_test_secret_token_123")
	}
}

func TestFetch_SkipIfExists(t *testing.T) {
	var requestCount int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&requestCount, 1)
		_, _ = w.Write([]byte("fresh server data"))
	}))
	defer srv.Close()

	tmpDir := t.TempDir()
	ctx := context.Background()

	dest := filepath.Join(tmpDir, "cached.litertlm")
	if err := os.WriteFile(dest, []byte("existing local cached data"), 0o644); err != nil {
		t.Fatalf("write cache: %v", err)
	}

	filePath, err := Fetch(ctx, srv.URL+"/cached.litertlm",
		WithDir(tmpDir),
		WithSkipIfExists(true),
	)
	if err != nil {
		t.Fatalf("Fetch: %v", err)
	}

	if atomic.LoadInt32(&requestCount) != 0 {
		t.Errorf("requestCount = %d, want 0 when skipIfExists is true", requestCount)
	}

	content, _ := os.ReadFile(filePath)
	if string(content) != "existing local cached data" {
		t.Errorf("content = %q, want existing local cached data", string(content))
	}
}

func TestFetch_TimeoutAndCancel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(500 * time.Millisecond)
		_, _ = w.Write([]byte("slow response"))
	}))
	defer srv.Close()

	tmpDir := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := Fetch(ctx, srv.URL+"/slow.litertlm", WithDir(tmpDir))
	if err == nil {
		t.Fatal("expected timeout/cancellation error, got nil")
	}
}
