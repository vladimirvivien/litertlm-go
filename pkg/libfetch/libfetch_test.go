package libfetch

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestPlatform(t *testing.T) {
	p, err := Platform()
	if runtime.GOOS == "windows" && runtime.GOARCH == "amd64" {
		if err != nil || p != "windows_x86_64" {
			t.Errorf("Platform() = %v, %v; want windows_x86_64, nil", p, err)
		}
	} else if runtime.GOOS == "linux" && runtime.GOARCH == "amd64" {
		if err != nil || p != "linux_x86_64" {
			t.Errorf("Platform() = %v, %v; want linux_x86_64, nil", p, err)
		}
	} else if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		if err != nil || p != "macos_arm64" {
			t.Errorf("Platform() = %v, %v; want macos_arm64, nil", p, err)
		}
	}
}

func TestPlatformFor(t *testing.T) {
	tests := []struct {
		goos, goarch string
		want         string
		wantErr      bool
	}{
		{"darwin", "arm64", "macos_arm64", false},
		{"darwin", "amd64", "macos_x86_64", false},
		{"linux", "amd64", "linux_x86_64", false},
		{"linux", "arm64", "linux_arm64", false},
		{"windows", "amd64", "windows_x86_64", false},
		{"plan9", "386", "", true},
	}
	for _, tc := range tests {
		got, err := PlatformFor(tc.goos, tc.goarch)
		if (err != nil) != tc.wantErr {
			t.Errorf("PlatformFor(%q, %q) error = %v, wantErr %v", tc.goos, tc.goarch, err, tc.wantErr)
		}
		if got != tc.want {
			t.Errorf("PlatformFor(%q, %q) = %q, want %q", tc.goos, tc.goarch, got, tc.want)
		}
	}
}

func TestDefaultDir(t *testing.T) {
	d, err := DefaultDir("v0.16.0")
	if err != nil {
		t.Fatalf("DefaultDir error: %v", err)
	}
	if !strings.Contains(d, "litertlm-go") || !strings.Contains(d, "v0.16.0") {
		t.Errorf("DefaultDir() = %q, expected path containing litertlm-go and v0.16.0", d)
	}
}

func TestWriteFile_Atomic(t *testing.T) {
	tmpDir := t.TempDir()
	dst := filepath.Join(tmpDir, "test.dll")
	data := []byte("mock binary content")

	if err := writeFile(dst, data); err != nil {
		t.Fatalf("writeFile failed: %v", err)
	}

	got, err := os.ReadFile(dst)
	if err != nil {
		t.Fatalf("ReadFile failed: %v", err)
	}
	if !bytes.Equal(got, data) {
		t.Errorf("ReadFile = %q, want %q", got, data)
	}
}

func TestFetch_MockedServer(t *testing.T) {
	libContent := []byte("mock shared library dll content")

	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	f1, err := zw.Create("libLiteRt.dll")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = f1.Write(libContent)
	_ = zw.Close()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/zip")
		_, _ = w.Write(buf.Bytes())
	}))
	defer server.Close()

	tmpDir := t.TempDir()
	c := config{
		version:  "v0.16.0",
		dir:      tmpDir,
		platform: "windows_x86_64",
		backend:  "cpu",
		logf:     t.Logf,
	}

	err = fetchAndExtractZip(context.Background(), &c, server.URL+"/archive.zip")
	if err != nil {
		t.Fatalf("fetchAndExtractZip failed: %v", err)
	}

	dst := filepath.Join(tmpDir, "libLiteRt.dll")
	got, err := os.ReadFile(dst)
	if err != nil {
		t.Fatalf("ReadFile error: %v", err)
	}
	if !bytes.Equal(got, libContent) {
		t.Errorf("got %q, want %q", got, libContent)
	}
}

func TestFetchDXC_Extraction(t *testing.T) {
	// Create mock zip containing bin/x64/dxcompiler.dll and dxil.dll
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)

	f1, err := zw.Create("bin/x64/dxcompiler.dll")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = f1.Write([]byte("mock dxcompiler"))

	f2, err := zw.Create("bin/x64/dxil.dll")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = f2.Write([]byte("mock dxil"))
	_ = zw.Close()

	zipData := buf.Bytes()
	zipHash := sha256.Sum256(zipData)
	zipHashHex := hex.EncodeToString(zipHash[:])

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(zipData)
	}))
	defer server.Close()

	tmpDir := t.TempDir()

	zr, err := zip.NewReader(bytes.NewReader(zipData), int64(len(zipData)))
	if err != nil {
		t.Fatal(err)
	}

	need := []string{"dxcompiler.dll", "dxil.dll"}
	for _, n := range need {
		entry := "bin/x64/" + n
		zf, err := zr.Open(entry)
		if err != nil {
			t.Fatalf("open entry %s: %v", entry, err)
		}
		content, _ := ioReadAll(zf)
		_ = zf.Close()
		if err := writeFile(filepath.Join(tmpDir, n), content); err != nil {
			t.Fatalf("write %s: %v", n, err)
		}
	}

	for _, n := range need {
		if _, err := os.Stat(filepath.Join(tmpDir, n)); err != nil {
			t.Errorf("expected %s to exist in tmpDir", n)
		}
	}
	_ = zipHashHex
}

func ioReadAll(r io.Reader) ([]byte, error) {
	var buf bytes.Buffer
	_, err := buf.ReadFrom(r)
	return buf.Bytes(), err
}
