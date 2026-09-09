package modelfetch

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// defaultHTTPClient is used when no custom client is provided via WithHTTPClient.
var defaultHTTPClient = &http.Client{
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

	target, err := ResolveModelIdentifierWithVariant(modelIDOrURL, c.variant)
	if err != nil {
		return "", err
	}

	c.filename = cmp.Or(c.filename, target.Filename)

	if c.dir == "" {
		d, dirErr := DefaultCacheDir()
		if dirErr != nil {
			return "", fmt.Errorf("modelfetch: resolve cache dir: %w", dirErr)
		}
		c.dir = d
	}

	if err = os.MkdirAll(c.dir, 0o755); err != nil {
		return "", fmt.Errorf("modelfetch: create target dir: %w", err)
	}

	destPath := filepath.Join(c.dir, c.filename)

	// Check if already downloaded or present in local model search dirs.
	if c.skipIfExists {
		if matchesLocalCandidate(destPath, c.sha256Sum) {
			return destPath, nil
		}
		if candidate := findLocalModel(c.filename); candidate != "" && matchesLocalCandidate(candidate, c.sha256Sum) {
			return candidate, nil
		}
	}

	authToken := cmp.Or(c.authToken, os.Getenv("HF_TOKEN"), os.Getenv("HUGGING_FACE_HUB_TOKEN"))

	if err = downloadFile(ctx, target.URL, destPath, authToken, &c); err != nil {
		return "", err
	}

	if c.sha256Sum != "" {
		match, sumErr := verifyChecksum(destPath, c.sha256Sum)
		if sumErr != nil {
			return "", fmt.Errorf("modelfetch: verify checksum: %w", sumErr)
		}
		if !match {
			_ = os.Remove(destPath)
			return "", fmt.Errorf("modelfetch: SHA-256 checksum mismatch for %s", destPath)
		}
	}

	return destPath, nil
}

func matchesLocalCandidate(path, sha256Sum string) bool {
	fi, err := os.Stat(path)
	if err != nil || fi.Size() <= 0 {
		return false
	}
	if sha256Sum == "" {
		return true
	}
	match, err := verifyChecksum(path, sha256Sum)
	return err == nil && match
}

func newDownloadRequest(ctx context.Context, targetURL, authToken string, offset int64) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, targetURL, nil)
	if err != nil {
		return nil, fmt.Errorf("modelfetch: create request: %w", err)
	}
	if authToken != "" {
		req.Header.Set("Authorization", "Bearer "+authToken)
	}
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}
	return req, nil
}

func downloadFile(ctx context.Context, url, destPath, authToken string, c *config) error {
	partPath := destPath + ".download"

	var existingSize int64
	if fi, err := os.Stat(partPath); err == nil {
		existingSize = fi.Size()
	}

	req, err := newDownloadRequest(ctx, url, authToken, existingSize)
	if err != nil {
		return err
	}

	client := cmp.Or(c.httpClient, defaultHTTPClient)
	if c.timeout > 0 {
		client = &http.Client{
			Timeout:   c.timeout,
			Transport: client.Transport,
		}
	}

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("modelfetch: HTTP request failed: %w", err)
	}
	defer func() {
		if resp != nil && resp.Body != nil {
			_ = resp.Body.Close()
		}
	}()

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
		_ = resp.Body.Close()
		resp = nil
		existingSize = 0
		retryReq, retryErr := newDownloadRequest(ctx, url, authToken, 0)
		if retryErr != nil {
			return retryErr
		}
		resp, err = client.Do(retryReq)
		if err != nil {
			return fmt.Errorf("modelfetch: retry request: %w", err)
		}
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("modelfetch: HTTP %s", resp.Status)
		}
		if resp.ContentLength > 0 {
			totalSize = resp.ContentLength
		}
		file, err = os.OpenFile(partPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
		if err != nil {
			return fmt.Errorf("modelfetch: open file: %w", err)
		}
	default:
		return fmt.Errorf("modelfetch: HTTP %s (%s)", resp.Status, url)
	}
	defer func() {
		if file != nil {
			_ = file.Close()
		}
	}()

	// Stream with progress
	pw := &progressWriter{
		writer:       file,
		downloaded:   existingSize,
		total:        totalSize,
		onProgress:   c.progress,
		lastReportAt: time.Now(),
	}

	pw.report()

	if _, err = io.Copy(pw, resp.Body); err != nil {
		return fmt.Errorf("modelfetch: streaming download: %w", err)
	}

	pw.report()

	if err = file.Close(); err != nil {
		file = nil
		return fmt.Errorf("modelfetch: close file: %w", err)
	}
	file = nil

	if err = os.Rename(partPath, destPath); err != nil {
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
		pw.report()
		pw.lastReportAt = time.Now()
	}
	return n, err
}

func (pw *progressWriter) report() {
	if pw.onProgress == nil {
		return
	}
	pct := -1.0
	if pw.total > 0 {
		pct = min((float64(pw.downloaded)/float64(pw.total))*100.0, 100.0)
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
	if _, err = io.Copy(h, f); err != nil {
		return false, err
	}

	gotHex := hex.EncodeToString(h.Sum(nil))
	return strings.EqualFold(gotHex, expectedHex), nil
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
