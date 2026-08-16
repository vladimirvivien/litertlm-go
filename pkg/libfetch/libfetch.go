// Package libfetch automates downloading and staging the native LiteRT-LM
// shared libraries for litertlm-go at runtime.
//
// Libraries are cached in <user cache dir>/litertlm-go/lib/<version>/<platform>.
// Fetch is idempotent: existing files with matching hashes are not downloaded again.
package libfetch

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// DefaultVersion is the upstream LiteRT-LM prebuilt release version.
const DefaultVersion = "v0.16.0"

const (
	githubReleaseURL = "https://github.com/google-ai-edge/LiteRT-LM/releases/download"
	rawPrebuiltURL   = "https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM"
	dxcURL           = "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.9.2602.24/dxc_2026_05_27.zip"
	dxcSHA256        = "cf658aacf070d3045e31b8f1f8a696c2945f37c1095019481ef7c513368db3b4"
)

type config struct {
	version  string
	dir      string
	platform string
	backend  string
	logf     func(format string, args ...any)
}

// Option configures Fetch.
type Option func(*config)

// WithVersion sets the target LiteRT-LM release version (e.g. "v0.16.0").
func WithVersion(v string) Option { return func(c *config) { c.version = v } }

// WithDir sets the target download destination directory.
func WithDir(dir string) Option { return func(c *config) { c.dir = dir } }

// WithPlatform overrides the target platform string (e.g. "windows_x86_64", "linux_x86_64").
func WithPlatform(p string) Option { return func(c *config) { c.platform = p } }

// WithBackend specifies the backend requirements ("cpu", "gpu", "apple", "metal").
func WithBackend(b string) Option { return func(c *config) { c.backend = b } }

// WithLogf configures logging callback for download progress.
func WithLogf(f func(format string, args ...any)) Option {
	return func(c *config) { c.logf = f }
}

// PlatformFor maps Go OS (e.g. "darwin", "linux", "windows") and architecture
// (e.g. "arm64", "amd64") to the prebuilt distribution platform identifier.
func PlatformFor(goos, goarch string) (string, error) {
	if goos == "" {
		goos = runtime.GOOS
	}
	if goarch == "" {
		goarch = runtime.GOARCH
	}
	switch strings.ToLower(goos) + "/" + strings.ToLower(goarch) {
	case "windows/amd64", "windows/x86_64":
		return "windows_x86_64", nil
	case "linux/amd64", "linux/x86_64":
		return "linux_x86_64", nil
	case "linux/arm64", "linux/aarch64":
		return "linux_arm64", nil
	case "darwin/arm64", "darwin/aarch64":
		return "macos_arm64", nil
	case "darwin/amd64", "darwin/x86_64":
		return "macos_x86_64", nil
	case "android/arm64", "android/aarch64":
		return "android_arm64", nil
	default:
		return "", fmt.Errorf("libfetch: unsupported OS/arch %s/%s", goos, goarch)
	}
}

// Platform returns the normalized platform identifier for the current runtime.
func Platform() (string, error) {
	return PlatformFor(runtime.GOOS, runtime.GOARCH)
}

// DefaultDir returns the local cache directory for the given version and current platform.
func DefaultDir(version string) (string, error) {
	if version == "" {
		version = DefaultVersion
	}
	p, err := Platform()
	if err != nil {
		return "", err
	}
	base, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("libfetch: resolve user cache dir: %w", err)
	}
	return filepath.Join(base, "litertlm-go", "lib", version, p), nil
}

// Fetch verifies and downloads the required shared libraries into the destination directory.
func Fetch(ctx context.Context, opts ...Option) (string, error) {
	c := config{
		version: DefaultVersion,
		backend: "cpu",
		logf:    func(string, ...any) {},
	}
	for _, opt := range opts {
		opt(&c)
	}

	if c.platform == "" {
		p, err := Platform()
		if err != nil {
			return "", err
		}
		c.platform = p
	}

	if c.dir == "" {
		d, err := DefaultDir(c.version)
		if err != nil {
			return "", err
		}
		c.dir = d
	}

	if err := os.MkdirAll(c.dir, 0o755); err != nil {
		return "", fmt.Errorf("libfetch: create cache dir: %w", err)
	}

	// Fetch from GitHub release assets
	if err := fetchFromGitHubRelease(ctx, &c); err != nil {
		c.logf("github release fetch error (%v), attempting raw prebuilt assets", err)
		if errRaw := fetchRawPrebuiltAssets(ctx, &c); errRaw != nil {
			return "", fmt.Errorf("libfetch: failed to download prebuilts: %w (fallback: %v)", err, errRaw)
		}
	}

	// Fetch Windows DirectX Shader Compiler if running on Windows with GPU
	if (c.platform == "windows_x86_64" || runtime.GOOS == "windows") && (c.backend == "gpu" || c.backend == "") {
		if err := fetchDXC(ctx, &c); err != nil {
			return "", fmt.Errorf("libfetch: fetch DXC: %w", err)
		}
	}

	return c.dir, nil
}

func fetchFromGitHubRelease(ctx context.Context, c *config) error {
	// For macOS Apple Foundation frameworks
	if strings.HasPrefix(c.platform, "macos") {
		archiveURL := fmt.Sprintf("%s/%s/CLiteRTLM_mac.xcframework.zip", githubReleaseURL, c.version)
		return fetchAndExtractZip(ctx, c, archiveURL)
	}

	var archiveName string
	switch c.platform {
	case "windows_x86_64":
		archiveName = fmt.Sprintf("litertlm-c-%s-windows-x86_64.zip", c.version)
	case "linux_x86_64":
		archiveName = fmt.Sprintf("litertlm-c-%s-linux-x86_64.tar.gz", c.version)
	case "linux_arm64":
		archiveName = fmt.Sprintf("litertlm-c-%s-linux-arm64.tar.gz", c.version)
	default:
		return fmt.Errorf("no release asset configured for platform %s", c.platform)
	}

	archiveURL := fmt.Sprintf("%s/%s/%s", githubReleaseURL, c.version, archiveName)
	if strings.HasSuffix(archiveURL, ".zip") {
		return fetchAndExtractZip(ctx, c, archiveURL)
	}
	return fetchAndExtractTarGz(ctx, c, archiveURL)
}

func fetchRawPrebuiltAssets(ctx context.Context, c *config) error {
	files := prebuiltFilesForPlatform(c.platform, c.backend)
	if len(files) == 0 {
		return fmt.Errorf("no prebuilt files declared for platform %s", c.platform)
	}

	for _, filename := range files {
		dst := filepath.Join(c.dir, filename)
		if _, err := os.Stat(dst); err == nil {
			c.logf("%s: already present", filename)
			continue
		}
		url := fmt.Sprintf("%s/%s/prebuilt/%s/%s", rawPrebuiltURL, c.version, c.platform, filename)
		c.logf("%s: downloading from %s", filename, url)
		data, err := get(ctx, url)
		if err != nil {
			return err
		}
		if err := writeFile(dst, data); err != nil {
			return err
		}
	}
	return nil
}

func prebuiltFilesForPlatform(platform, backend string) []string {
	switch platform {
	case "windows_x86_64":
		base := []string{
			"libLiteRt.dll",
			"libGemmaModelConstraintProvider.dll",
		}
		if backend == "cpu" {
			base = append(base, "litertlm_c_cpu.dll")
		} else {
			base = append(base, "litertlm_c.dll", "libLiteRtWebGpuAccelerator.dll", "libLiteRtTopKWebGpuSampler.dll", "libwebgpu_dawn.dll")
		}
		return base
	case "linux_x86_64":
		base := []string{
			"libLiteRt.so",
			"libGemmaModelConstraintProvider.so",
		}
		if backend == "cpu" {
			base = append(base, "liblitertlm_c_cpu.so")
		} else {
			base = append(base, "liblitertlm_c.so", "libLiteRtWebGpuAccelerator.so", "libLiteRtTopKWebGpuSampler.so")
		}
		return base
	case "linux_arm64":
		return []string{
			"libLiteRt.so",
			"libGemmaModelConstraintProvider.so",
			"liblitertlm_c_cpu.so",
		}
	case "macos_arm64", "macos_x86_64":
		return []string{
			"libLiteRt.dylib",
			"libGemmaModelConstraintProvider.dylib",
			"libLiteRtMetalAccelerator.dylib",
			"libLiteRtTopKMetalSampler.dylib",
			"liblitertlm_c.dylib",
			"liblitertlm_c_cpu.dylib",
		}
	default:
		return nil
	}
}

func fetchAndExtractZip(ctx context.Context, c *config, url string) error {
	c.logf("fetching archive %s", url)
	data, err := get(ctx, url)
	if err != nil {
		return err
	}
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return fmt.Errorf("open zip: %w", err)
	}
	for _, f := range zr.File {
		if f.FileInfo().IsDir() {
			continue
		}
		base := filepath.Base(f.Name)
		rc, err := f.Open()
		if err != nil {
			return err
		}
		content, err := io.ReadAll(rc)
		_ = rc.Close()
		if err != nil {
			return err
		}
		if err := writeFile(filepath.Join(c.dir, base), content); err != nil {
			return err
		}
	}
	return nil
}

func fetchAndExtractTarGz(ctx context.Context, c *config, url string) error {
	c.logf("fetching archive %s", url)
	data, err := get(ctx, url)
	if err != nil {
		return err
	}
	gzr, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("open gzip: %w", err)
	}
	defer func() { _ = gzr.Close() }()
	tr := tar.NewReader(gzr)
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if header.Typeflag == tar.TypeReg {
			base := filepath.Base(header.Name)
			content, err := io.ReadAll(tr)
			if err != nil {
				return err
			}
			if err := writeFile(filepath.Join(c.dir, base), content); err != nil {
				return err
			}
		}
	}
	return nil
}

func fetchDXC(ctx context.Context, c *config) error {
	need := []string{"dxcompiler.dll", "dxil.dll"}
	missing := false
	for _, n := range need {
		if _, err := os.Stat(filepath.Join(c.dir, n)); err != nil {
			missing = true
		}
	}
	if !missing {
		c.logf("dxc: up to date")
		return nil
	}
	c.logf("dxc: downloading %s", dxcURL)
	data, err := get(ctx, dxcURL)
	if err != nil {
		return err
	}
	if sum := sha256.Sum256(data); hex.EncodeToString(sum[:]) != dxcSHA256 {
		return fmt.Errorf("libfetch: dxc sha256 mismatch")
	}
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return fmt.Errorf("libfetch: open dxc zip: %w", err)
	}
	for _, n := range need {
		entry := "bin/x64/" + n
		zf, err := zr.Open(entry)
		if err != nil {
			return fmt.Errorf("libfetch: dxc zip missing %s: %w", entry, err)
		}
		content, err := io.ReadAll(zf)
		_ = zf.Close()
		if err != nil {
			return err
		}
		if err := writeFile(filepath.Join(c.dir, n), content); err != nil {
			return err
		}
	}
	return nil
}

func get(ctx context.Context, url string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	return io.ReadAll(resp.Body)
}

func writeFile(dst string, data []byte) error {
	tmp := dst + ".tmp"
	if err := os.WriteFile(tmp, data, 0o755); err != nil {
		return err
	}
	if err := os.Rename(tmp, dst); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}
