package modelfetch

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// DefaultHTTPClient is the standard HTTP client used if none is overridden.
var DefaultHTTPClient = &http.Client{
	Timeout: 0, // No global timeout by default for large model downloads; use context timeout.
}

// Fetch resolves and downloads a .litertlm model file to a local destination directory.
// It supports chunked download resuming, progress reporting, and auth tokens.
func Fetch(ctx context.Context, modelIDOrURL string, opts ...Option) (string, error) {
	c := config{
		skipIfExists: true,
		timeout:      0,
		progress:     func(int64, int64, float64) {},
	}
	for _, opt := range opts {
		opt(&c)
	}

	target, err := ResolveModelIdentifier(modelIDOrURL)
	if err != nil {
		return "", err
	}

	if c.filename == "" {
		c.filename = target.Filename
	}

	if c.dir == "" {
		d, err := DefaultCacheDir()
		if err != nil {
			return "", fmt.Errorf("modelfetch: resolve cache dir: %w", err)
		}
		c.dir = d
	}

	if err := os.MkdirAll(c.dir, 0o755); err != nil {
		return "", fmt.Errorf("modelfetch: create target dir: %w", err)
	}

	destPath := filepath.Join(c.dir, c.filename)

	// Check if already downloaded
	if c.skipIfExists {
		if fi, err := os.Stat(destPath); err == nil && fi.Size() > 0 {
			if c.sha256Sum != "" {
				if match, err := verifyChecksum(destPath, c.sha256Sum); err == nil && match {
					return destPath, nil
				}
			} else {
				return destPath, nil
			}
		}

		// Also check local candidate directories (e.g. ~/models, LITERTLM_MODELS_DIR)
		if candidate := findLocalModel(c.filename); candidate != "" {
			return candidate, nil
		}
	}

	// Resolve Bearer Token
	authToken := c.authToken
	if authToken == "" {
		authToken = os.Getenv("HF_TOKEN")
		if authToken == "" {
			authToken = os.Getenv("HUGGING_FACE_HUB_TOKEN")
		}
	}

	if err := downloadFile(ctx, target.URL, destPath, authToken, &c); err != nil {
		return "", err
	}

	if c.sha256Sum != "" {
		match, err := verifyChecksum(destPath, c.sha256Sum)
		if err != nil {
			return "", fmt.Errorf("modelfetch: verify checksum: %w", err)
		}
		if !match {
			_ = os.Remove(destPath)
			return "", fmt.Errorf("modelfetch: SHA-256 checksum mismatch for %s", destPath)
		}
	}

	return destPath, nil
}

func downloadFile(ctx context.Context, url, destPath, authToken string, c *config) error {
	partPath := destPath + ".download"

	var existingSize int64
	if fi, err := os.Stat(partPath); err == nil {
		existingSize = fi.Size()
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("modelfetch: create request: %w", err)
	}

	if authToken != "" {
		req.Header.Set("Authorization", "Bearer "+authToken)
	}

	if existingSize > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", existingSize))
	}

	client := DefaultHTTPClient
	if c.timeout > 0 {
		client = &http.Client{Timeout: c.timeout}
	}

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("modelfetch: HTTP request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	var file *os.File
	var totalSize int64 = -1

	switch resp.StatusCode {
	case http.StatusOK:
		// Full download (server does not support Range or fresh file)
		existingSize = 0
		if resp.ContentLength > 0 {
			totalSize = resp.ContentLength
		}
		file, err = os.OpenFile(partPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
		if err != nil {
			return fmt.Errorf("modelfetch: open file: %w", err)
		}
	case http.StatusPartialContent:
		// Range resume supported
		if resp.ContentLength > 0 {
			totalSize = existingSize + resp.ContentLength
		}
		file, err = os.OpenFile(partPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return fmt.Errorf("modelfetch: open file for resume: %w", err)
		}
	case http.StatusRequestedRangeNotSatisfiable:
		// File might already be complete
		existingSize = 0
		req.Header.Del("Range")
		resp2, err := client.Do(req)
		if err != nil {
			return fmt.Errorf("modelfetch: retry request: %w", err)
		}
		defer func() { _ = resp2.Body.Close() }()
		if resp2.StatusCode != http.StatusOK {
			return fmt.Errorf("modelfetch: HTTP %s", resp2.Status)
		}
		if resp2.ContentLength > 0 {
			totalSize = resp2.ContentLength
		}
		file, err = os.OpenFile(partPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
		if err != nil {
			return fmt.Errorf("modelfetch: open file: %w", err)
		}
		resp = resp2
	default:
		return fmt.Errorf("modelfetch: HTTP %s (%s)", resp.Status, url)
	}
	defer func() { _ = file.Close() }()

	// Stream with progress
	pw := &progressWriter{
		writer:       file,
		downloaded:   existingSize,
		total:        totalSize,
		onProgress:   c.progress,
		lastReportAt: time.Now(),
	}

	// Initial report
	pw.report(false)

	if _, err := io.Copy(pw, resp.Body); err != nil {
		return fmt.Errorf("modelfetch: streaming download: %w", err)
	}

	// Final 100% progress report
	pw.report(true)

	if err := file.Close(); err != nil {
		return fmt.Errorf("modelfetch: close file: %w", err)
	}

	// Atomic rename
	if err := os.Rename(partPath, destPath); err != nil {
		return fmt.Errorf("modelfetch: finalize model file: %w", err)
	}

	return nil
}

type progressWriter struct {
	writer       io.Writer
	downloaded   int64
	total        int64
	onProgress   ProgressFunc
	lastReportAt time.Time
}

func (pw *progressWriter) Write(p []byte) (int, error) {
	n, err := pw.writer.Write(p)
	pw.downloaded += int64(n)
	if time.Since(pw.lastReportAt) > 100*time.Millisecond {
		pw.report(false)
		pw.lastReportAt = time.Now()
	}
	return n, err
}

func (pw *progressWriter) report(force bool) {
	if pw.onProgress == nil {
		return
	}
	pct := -1.0
	if pw.total > 0 {
		pct = (float64(pw.downloaded) / float64(pw.total)) * 100.0
		if pct > 100.0 {
			pct = 100.0
		}
	}
	pw.onProgress(pw.downloaded, pw.total, pct)
}

func verifyChecksum(filePath, expectedHex string) (bool, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return false, err
	}
	defer func() { _ = f.Close() }()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return false, err
	}

	gotHex := hex.EncodeToString(h.Sum(nil))
	return gotHex == expectedHex, nil
}

func findLocalModel(filename string) string {
	var candidates []string
	if env := os.Getenv("LITERTLM_MODELS_DIR"); env != "" {
		candidates = append(candidates, filepath.Join(env, filename))
	}
	if home, err := os.UserHomeDir(); err == nil {
		candidates = append(candidates,
			filepath.Join(home, "models", filename),
			filepath.Join(home, ".litertlm", "models", filename),
			filepath.Join(home, "Downloads", filename),
		)
	}
	for _, c := range candidates {
		if fi, err := os.Stat(c); err == nil && fi.Size() > 0 {
			return c
		}
	}
	return ""
}
