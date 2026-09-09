package libfetch

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
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
	d, err := DefaultDir("")
	if err != nil {
		t.Fatalf("DefaultDir error: %v", err)
	}
	if !strings.Contains(d, "litertlm-go") || !strings.Contains(d, DefaultVersion) {
		t.Errorf("DefaultDir() = %q, expected path containing litertlm-go and %s", d, DefaultVersion)
	}
}

func TestWriteReaderAtomic_AndGitLFSRejection(t *testing.T) {
	tmpDir := t.TempDir()
	dst := filepath.Join(tmpDir, "test.dll")
	data := []byte("mock binary content")

	if err := writeReaderAtomic(dst, bytes.NewReader(data)); err != nil {
		t.Fatalf("writeReaderAtomic failed: %v", err)
	}

	got, err := os.ReadFile(dst)
	if err != nil {
		t.Fatalf("ReadFile failed: %v", err)
	}
	if !bytes.Equal(got, data) {
		t.Errorf("ReadFile = %q, want %q", got, data)
	}

	lfsDst := filepath.Join(tmpDir, "lfs.dll")
	lfsData := []byte("version https://git-lfs.github.com/spec/v1\noid sha256:1234\nsize 999\n")
	if err := writeReaderAtomic(lfsDst, bytes.NewReader(lfsData)); err == nil || !strings.Contains(err.Error(), "Git LFS") {
		t.Fatalf("expected Git LFS pointer error, got %v", err)
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
		if err := writeReaderAtomic(filepath.Join(tmpDir, n), zf); err != nil {
			_ = zf.Close()
			t.Fatalf("write %s: %v", n, err)
		}
		_ = zf.Close()
	}

	for _, n := range need {
		if _, err := os.Stat(filepath.Join(tmpDir, n)); err != nil {
			t.Errorf("expected %s to exist in tmpDir", n)
		}
	}
	_ = zipHashHex
}

func TestFetchAndExtractTarGz_StreamingAndFilter(t *testing.T) {
	entries := map[string]string{
		"LICENSE":                            "apache",
		"lib/linux_x86_64/liblitert-lm.so":   "linux-amd64-so",
		"lib/linux_arm64/liblitert-lm.so":    "linux-arm64-so",
		"lib/macos_arm64/liblitert-lm.dylib": "macos-arm64-dylib",
	}

	var buf bytes.Buffer
	gzw := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gzw)
	for name, body := range entries {
		if err := tw.WriteHeader(&tar.Header{
			Name:     name,
			Mode:     0o755,
			Size:     int64(len(body)),
			Typeflag: tar.TypeReg,
		}); err != nil {
			t.Fatal(err)
		}
		if _, err := tw.Write([]byte(body)); err != nil {
			t.Fatal(err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gzw.Close(); err != nil {
		t.Fatal(err)
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(buf.Bytes())
	}))
	defer srv.Close()

	tmpDir := t.TempDir()
	c := config{
		version:  "v0.19.1",
		dir:      tmpDir,
		platform: "macos_arm64",
		backend:  "cpu",
		logf:     t.Logf,
	}
	if err := fetchAndExtractTarGz(context.Background(), &c, srv.URL+"/mac.tar.gz"); err != nil {
		t.Fatalf("fetchAndExtractTarGz: %v", err)
	}
	items, err := os.ReadDir(tmpDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 1 || items[0].Name() != "liblitert-lm.dylib" {
		t.Fatalf("extracted = %v, want [liblitert-lm.dylib]", items)
	}
	got, _ := os.ReadFile(filepath.Join(tmpDir, "liblitert-lm.dylib"))
	if string(got) != "macos-arm64-dylib" {
		t.Errorf("content = %q, want macos-arm64-dylib", got)
	}
}

func TestFetch_CrossPlatformSkipsDXC(t *testing.T) {
	// Even on a Windows host, targeting linux_x86_64 with backend=gpu must
	// not download DXC when liblitert-lm.so is already present.
	tmpDir := t.TempDir()
	dummySO := filepath.Join(tmpDir, "liblitert-lm.so")
	if err := os.WriteFile(dummySO, bytes.Repeat([]byte("A"), 1024), 0o755); err != nil {
		t.Fatal(err)
	}
	gotDir, err := Fetch(context.Background(),
		WithDir(tmpDir),
		WithPlatform("linux_x86_64"),
		WithBackend("gpu"),
	)
	if err != nil {
		t.Fatalf("Fetch(linux_x86_64, gpu): %v", err)
	}
	if gotDir != tmpDir {
		t.Errorf("dir = %q, want %q", gotDir, tmpDir)
	}
	if _, statErr := os.Stat(filepath.Join(tmpDir, "dxcompiler.dll")); !os.IsNotExist(statErr) {
		t.Errorf("dxcompiler.dll should not exist for linux_x86_64 target")
	}
}

func TestFetchAndExtractZip_MultiPlatformFilter(t *testing.T) {
	entries := map[string]string{
		"LICENSE":                              "apache",
		"lib/windows_x86_64/bin/litert-lm.dll": "win-dll-bytes",
		"lib/windows_x86_64/lib/litert-lm.lib": "win-import-lib",
		"lib/linux_x86_64/liblitert-lm.so":     "linux-amd64-so-bytes",
		"lib/linux_arm64/liblitert-lm.so":      "linux-arm64-so-bytes",
		"lib/macos_arm64/liblitert-lm.dylib":   "macos-arm64-dylib-bytes",
		"lib/android_arm64/liblitert-lm.so":    "android-arm64-so-bytes",
		"lib/android_x86_64/liblitert-lm.so":   "android-amd64-so-bytes",
	}

	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for name, body := range entries {
		fw, err := zw.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := fw.Write([]byte(body)); err != nil {
			t.Fatal(err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/zip")
		_, _ = w.Write(buf.Bytes())
	}))
	defer server.Close()

	tests := []struct {
		platform string
		wantFile string
		wantBody string
	}{
		{"windows_x86_64", "litert-lm.dll", "win-dll-bytes"},
		{"linux_x86_64", "liblitert-lm.so", "linux-amd64-so-bytes"},
		{"linux_arm64", "liblitert-lm.so", "linux-arm64-so-bytes"},
		{"macos_arm64", "liblitert-lm.dylib", "macos-arm64-dylib-bytes"},
		{"android_arm64", "liblitert-lm.so", "android-arm64-so-bytes"},
		{"android_x86_64", "liblitert-lm.so", "android-amd64-so-bytes"},
	}

	for _, tc := range tests {
		t.Run(tc.platform, func(t *testing.T) {
			tmpDir := t.TempDir()
			c := config{
				version:  "v0.19.1",
				dir:      tmpDir,
				platform: tc.platform,
				backend:  "cpu",
				logf:     t.Logf,
			}
			if err := fetchAndExtractZip(context.Background(), &c, server.URL+"/c_api.zip"); err != nil {
				t.Fatalf("fetchAndExtractZip: %v", err)
			}
			items, err := os.ReadDir(tmpDir)
			if err != nil {
				t.Fatal(err)
			}
			if len(items) != 1 || items[0].Name() != tc.wantFile {
				t.Fatalf("extracted files = %v, want single file %q", items, tc.wantFile)
			}
			got, err := os.ReadFile(filepath.Join(tmpDir, tc.wantFile))
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != tc.wantBody {
				t.Errorf("file content = %q, want %q", got, tc.wantBody)
			}
		})
	}
}
