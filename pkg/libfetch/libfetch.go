// Package libfetch automates downloading and staging the native LiteRT-LM
// shared libraries for litertlm-go at runtime.
//
// Libraries are cached in <user cache dir>/litertlm-go/lib/<version>/<platform>.
// Fetch is idempotent: existing unified libraries are reused without re-downloading.
package libfetch

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"cmp"
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
const DefaultVersion = "v0.19.1"

const (
	githubReleaseURL = "https://github.com/google-ai-edge/LiteRT-LM/releases/download"
	rawPrebuiltURL   = "https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM"
	dxcURL           = "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.9.2602.24/dxc_2026_05_27.zip"
	dxcSHA256        = "cf658aacf070d3045e31b8f1f8a696c2945f37c1095019481ef7c513368db3b4"
)

var lfsPrefix = []byte("version https://git-lfs.github.com/spec/v1")

type config struct {
	version  string
	dir      string
	platform string
	backend  string
	logf     func(format string, args ...any)
}

// Option configures Fetch.
type Option func(*config)

// WithVersion sets the target LiteRT-LM release version (e.g. "v0.19.1").
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
	goos = cmp.Or(goos, runtime.GOOS)
	goarch = cmp.Or(goarch, runtime.GOARCH)
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
	case "android/amd64", "android/x86_64":
		return "android_x86_64", nil
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
	version = cmp.Or(version, DefaultVersion)
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

	if hasUnifiedLib(c.dir, c.platform) {
		c.logf("%s: already present", unifiedLibFilename(c.platform))
	} else {
		if err := fetchFromGitHubRelease(ctx, &c); err != nil {
			c.logf("github release fetch error (%v), attempting raw prebuilt assets", err)
			if errRaw := fetchRawPrebuiltAssets(ctx, &c); errRaw != nil {
				return "", fmt.Errorf("libfetch: download prebuilts for %s (%s): %w (fallback: %w); if this release has not published cross-platform C API archives yet, use v0.19.1 or build //c:litert-lm from source", c.version, c.platform, err, errRaw)
			}
		}
	}

	// Fetch Windows DirectX Shader Compiler only when targeting Windows GPU.
	if c.platform == "windows_x86_64" && (c.backend == "gpu" || c.backend == "") {
		if err := fetchDXC(ctx, &c); err != nil {
			return "", fmt.Errorf("libfetch: fetch DXC: %w", err)
		}
	}

	// Skip copying legacy auxiliary DLLs into a directory that already holds the unified C API 1.0+ library.
	if !hasUnifiedLib(c.dir, c.platform) {
		populateFromLocalStaging(&c)
	}

	return c.dir, nil
}

func unifiedLibFilename(platform string) string {
	switch platform {
	case "windows_x86_64":
		return "litert-lm.dll"
	case "macos_arm64", "macos_x86_64":
		return "liblitert-lm.dylib"
	case "linux_x86_64", "linux_arm64", "android_arm64", "android_x86_64":
		return "liblitert-lm.so"
	default:
		return ""
	}
}

func hasUnifiedLib(dir, platform string) bool {
	name := unifiedLibFilename(platform)
	if name == "" {
		return false
	}
	info, err := os.Stat(filepath.Join(dir, name))
	return err == nil && info.Size() > 512
}

func populateFromLocalStaging(c *config) {
	var candidateDirs []string
	if env := os.Getenv("LITERTLM_LIB"); env != "" {
		candidateDirs = append(candidateDirs, env)
	}
	if home, err := os.UserHomeDir(); err == nil {
		candidateDirs = append(candidateDirs,
			filepath.Join(home, "include", "litertlm", "lib"),
			filepath.Join(home, ".litertlm", "lib"),
		)
	}
	candidateDirs = append(candidateDirs, "/opt/litertlm/lib", "/usr/local/lib")

	for _, dir := range candidateDirs {
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			dst := filepath.Join(c.dir, entry.Name())
			if _, statErr := os.Stat(dst); statErr == nil {
				continue
			}
			info, infoErr := entry.Info()
			if infoErr != nil || info.Size() <= 512 {
				continue
			}
			f, openErr := os.Open(filepath.Join(dir, entry.Name()))
			if openErr != nil {
				continue
			}
			_ = writeReaderAtomic(dst, f)
			_ = f.Close()
		}
	}
}

func fetchFromGitHubRelease(ctx context.Context, c *config) error {
	trimmedVer := strings.TrimPrefix(c.version, "v")
	candidateURLs := []string{
		fmt.Sprintf("%s/%s/litert_lm_c_api-1.1.0.zip", githubReleaseURL, c.version),
		fmt.Sprintf("%s/%s/litert_lm_c_api-1.0.0.zip", githubReleaseURL, c.version),
		fmt.Sprintf("%s/%s/litert_lm_c_api-%s.zip", githubReleaseURL, c.version, trimmedVer),
	}

	switch c.platform {
	case "macos_arm64":
		candidateURLs = append(candidateURLs,
			fmt.Sprintf("%s/%s/litert_lm_macos_arm64.tar.gz", githubReleaseURL, c.version),
			fmt.Sprintf("%s/%s/CLiteRTLM_mac.xcframework.zip", githubReleaseURL, c.version),
			fmt.Sprintf("%s/%s/CLiteRTLM.xcframework.zip", githubReleaseURL, c.version),
		)
	case "macos_x86_64":
		candidateURLs = append(candidateURLs,
			fmt.Sprintf("%s/%s/CLiteRTLM_mac.xcframework.zip", githubReleaseURL, c.version),
			fmt.Sprintf("%s/%s/CLiteRTLM.xcframework.zip", githubReleaseURL, c.version),
		)
	case "windows_x86_64":
		candidateURLs = append(candidateURLs,
			fmt.Sprintf("%s/%s/litertlm-c-%s-windows-x86_64.zip", githubReleaseURL, c.version, c.version),
			fmt.Sprintf("%s/%s/litertlm-windows-x86_64-%s.zip", githubReleaseURL, c.version, c.version),
		)
	case "linux_x86_64":
		candidateURLs = append(candidateURLs,
			fmt.Sprintf("%s/%s/litertlm-c-%s-linux-x86_64.tar.gz", githubReleaseURL, c.version, c.version),
			fmt.Sprintf("%s/%s/litertlm-linux-x86_64-%s.tar.gz", githubReleaseURL, c.version, c.version),
		)
	case "linux_arm64":
		candidateURLs = append(candidateURLs,
			fmt.Sprintf("%s/%s/litertlm-c-%s-linux-arm64.tar.gz", githubReleaseURL, c.version, c.version),
			fmt.Sprintf("%s/%s/litertlm-linux-arm64-%s.tar.gz", githubReleaseURL, c.version, c.version),
		)
	}

	var lastErr error
	for _, archiveURL := range candidateURLs {
		if err := ctx.Err(); err != nil {
			return err
		}
		if strings.HasSuffix(archiveURL, ".zip") {
			if err := fetchAndExtractZip(ctx, c, archiveURL); err == nil {
				return nil
			} else {
				lastErr = err
			}
		} else {
			if err := fetchAndExtractTarGz(ctx, c, archiveURL); err == nil {
				return nil
			} else {
				lastErr = err
			}
		}
	}

	if lastErr != nil {
		return lastErr
	}
	return fmt.Errorf("no valid release asset found for platform %s", c.platform)
}

func fetchRawPrebuiltAssets(ctx context.Context, c *config) error {
	files := prebuiltFilesForPlatform(c.platform, c.backend)
	if len(files) == 0 {
		return fmt.Errorf("no prebuilt files declared for platform %s", c.platform)
	}

	for _, filename := range files {
		if err := ctx.Err(); err != nil {
			return err
		}
		dst := filepath.Join(c.dir, filename)
		if _, err := os.Stat(dst); err == nil {
			c.logf("%s: already present", filename)
			continue
		}
		url := fmt.Sprintf("%s/%s/prebuilt/%s/%s", rawPrebuiltURL, c.version, c.platform, filename)
		c.logf("%s: downloading from %s", filename, url)
		if err := downloadFileAtomic(ctx, url, dst); err != nil {
			return err
		}
	}
	return nil
}

func downloadFileAtomic(ctx context.Context, url, dst string) error {
	resp, err := openHTTP(ctx, url)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	return writeReaderAtomic(dst, resp.Body)
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

func isLibraryForPlatform(platform, name string) bool {
	ext := strings.ToLower(filepath.Ext(name))
	switch platform {
	case "windows_x86_64":
		return ext == ".dll"
	case "macos_arm64", "macos_x86_64":
		return ext == ".dylib" || strings.Contains(name, ".framework")
	case "linux_x86_64", "linux_arm64", "android_arm64", "android_x86_64":
		return ext == ".so" || strings.Contains(name, ".so.")
	default:
		return ext == ".so" || ext == ".dylib" || ext == ".dll"
	}
}

func archiveEntryMatchesPlatform(platform, entryName string) (string, bool) {
	norm := strings.TrimPrefix(filepath.ToSlash(entryName), "./")
	if strings.HasPrefix(norm, "lib/") {
		platformPrefix := "lib/" + platform + "/"
		if !strings.HasPrefix(norm, platformPrefix) {
			return "", false
		}
	}
	base := filepath.Base(norm)
	if !isLibraryForPlatform(platform, base) {
		return "", false
	}
	return base, true
}

func fetchAndExtractZip(ctx context.Context, c *config, url string) error {
	c.logf("fetching archive %s", url)
	tmpZip, _, err := downloadToTempFile(ctx, url, c.dir, false)
	if err != nil {
		return err
	}
	zr, err := zip.OpenReader(tmpZip)
	if err != nil {
		_ = os.Remove(tmpZip)
		return fmt.Errorf("open zip: %w", err)
	}
	// Close zip reader before removing temp file (required on Windows).
	defer func() {
		_ = zr.Close()
		_ = os.Remove(tmpZip)
	}()

	extracted := 0
	for _, f := range zr.File {
		if f.FileInfo().IsDir() {
			continue
		}
		base, ok := archiveEntryMatchesPlatform(c.platform, f.Name)
		if !ok {
			continue
		}
		if err = extractZipEntry(filepath.Join(c.dir, base), f); err != nil {
			return err
		}
		extracted++
	}
	if extracted == 0 {
		return fmt.Errorf("no matching library files found in archive %s", url)
	}
	return nil
}

func extractZipEntry(dst string, f *zip.File) error {
	rc, err := f.Open()
	if err != nil {
		return err
	}
	defer func() { _ = rc.Close() }()
	return writeReaderAtomic(dst, rc)
}

func fetchAndExtractTarGz(ctx context.Context, c *config, url string) error {
	c.logf("fetching archive %s", url)
	resp, err := openHTTP(ctx, url)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	gzr, err := gzip.NewReader(resp.Body)
	if err != nil {
		return fmt.Errorf("open gzip: %w", err)
	}
	defer func() { _ = gzr.Close() }()

	tr := tar.NewReader(gzr)
	extracted := 0
	for {
		header, nextErr := tr.Next()
		if nextErr == io.EOF {
			break
		}
		if nextErr != nil {
			return nextErr
		}
		if header.Typeflag == tar.TypeReg {
			base, ok := archiveEntryMatchesPlatform(c.platform, header.Name)
			if !ok {
				continue
			}
			if err = writeReaderAtomic(filepath.Join(c.dir, base), tr); err != nil {
				return err
			}
			extracted++
		}
	}
	if extracted == 0 {
		return fmt.Errorf("no matching library files found in archive %s", url)
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
	tmpZip, sumHex, err := downloadToTempFile(ctx, dxcURL, c.dir, true)
	if err != nil {
		return err
	}
	if !strings.EqualFold(sumHex, dxcSHA256) {
		_ = os.Remove(tmpZip)
		return fmt.Errorf("libfetch: dxc sha256 mismatch")
	}
	zr, err := zip.OpenReader(tmpZip)
	if err != nil {
		_ = os.Remove(tmpZip)
		return fmt.Errorf("libfetch: open dxc zip: %w", err)
	}
	defer func() {
		_ = zr.Close()
		_ = os.Remove(tmpZip)
	}()

	for _, n := range need {
		entry := "bin/x64/" + n
		zf, openErr := zr.Open(entry)
		if openErr != nil {
			return fmt.Errorf("libfetch: dxc zip missing %s: %w", entry, openErr)
		}
		writeErr := writeReaderAtomic(filepath.Join(c.dir, n), zf)
		_ = zf.Close()
		if writeErr != nil {
			return writeErr
		}
	}
	return nil
}

func openHTTP(ctx context.Context, url string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		_ = resp.Body.Close()
		return nil, fmt.Errorf("GET %s: %s", url, resp.Status)
	}
	return resp, nil
}

func downloadToTempFile(ctx context.Context, url, dir string, verifyHash bool) (tmpPath, sumHex string, err error) {
	resp, err := openHTTP(ctx, url)
	if err != nil {
		return "", "", err
	}
	defer func() { _ = resp.Body.Close() }()

	tmp, err := os.CreateTemp(dir, "libfetch-archive-*.tmp")
	if err != nil {
		return "", "", err
	}
	tmpPath = tmp.Name()
	cleanup := true
	defer func() {
		if cleanup {
			_ = tmp.Close()
			_ = os.Remove(tmpPath)
		}
	}()

	var dst io.Writer = tmp
	h := sha256.New()
	if verifyHash {
		dst = io.MultiWriter(tmp, h)
	}
	if _, err = io.Copy(dst, resp.Body); err != nil {
		return "", "", err
	}
	if err = tmp.Close(); err != nil {
		return "", "", err
	}
	cleanup = false
	if verifyHash {
		sumHex = hex.EncodeToString(h.Sum(nil))
	}
	return tmpPath, sumHex, nil
}

func writeReaderAtomic(dst string, r io.Reader) error {
	head := make([]byte, len(lfsPrefix))
	n, err := io.ReadFull(r, head)
	if err != nil && err != io.EOF && err != io.ErrUnexpectedEOF {
		return err
	}
	head = head[:n]
	if bytes.Equal(head, lfsPrefix) {
		return fmt.Errorf("refusing to write Git LFS pointer for binary %s", filepath.Base(dst))
	}

	tmp, err := os.CreateTemp(filepath.Dir(dst), filepath.Base(dst)+".*.tmp")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	cleanup := true
	defer func() {
		_ = tmp.Close()
		if cleanup {
			_ = os.Remove(tmpPath)
		}
	}()

	if len(head) > 0 {
		if _, err = tmp.Write(head); err != nil {
			return err
		}
	}
	if _, err = io.Copy(tmp, r); err != nil {
		return err
	}
	if err = tmp.Chmod(0o755); err != nil {
		return err
	}
	if err = tmp.Close(); err != nil {
		return err
	}
	if err = os.Rename(tmpPath, dst); err != nil {
		return err
	}
	cleanup = false
	return nil
}
