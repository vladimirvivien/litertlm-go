package modelfetch

import (
	"os"
	"path/filepath"
	"time"
)

// ProgressFunc receives the number of downloaded bytes, total bytes (-1 if unknown),
// and the percentage complete (0.0 to 100.0, or -1.0 if unknown).
type ProgressFunc func(downloaded, total int64, pct float64)

type config struct {
	dir          string
	filename     string
	authToken    string
	progress     ProgressFunc
	timeout      time.Duration
	skipIfExists bool
	sha256Sum    string
}

// Option configures model download behavior.
type Option func(*config)

// WithDir sets the target destination directory where the model will be stored.
func WithDir(dir string) Option {
	return func(c *config) {
		c.dir = dir
	}
}

// WithFilename explicitly sets the destination filename.
func WithFilename(name string) Option {
	return func(c *config) {
		c.filename = name
	}
}

// WithAuthToken sets the Bearer authorization token (e.g. for gated Hugging Face models).
func WithAuthToken(token string) Option {
	return func(c *config) {
		c.authToken = token
	}
}

// WithProgress registers a callback invoked periodically during download with progress updates.
func WithProgress(fn ProgressFunc) Option {
	return func(c *config) {
		c.progress = fn
	}
}

// WithTimeout sets the HTTP client timeout for network operations.
func WithTimeout(d time.Duration) Option {
	return func(c *config) {
		c.timeout = d
	}
}

// WithSkipIfExists configures whether to return the existing file if it already exists and is non-empty.
func WithSkipIfExists(skip bool) Option {
	return func(c *config) {
		c.skipIfExists = skip
	}
}

// WithSHA256 sets the expected SHA-256 hex checksum to verify after download.
func WithSHA256(checksumHex string) Option {
	return func(c *config) {
		c.sha256Sum = checksumHex
	}
}

// DefaultCacheDir returns the default local cache directory for downloaded models.
func DefaultCacheDir() (string, error) {
	if env := os.Getenv("LITERTLM_MODELS_DIR"); env != "" {
		return env, nil
	}
	base, err := os.UserCacheDir()
	if err != nil {
		home, errHome := os.UserHomeDir()
		if errHome != nil {
			return "", err
		}
		return filepath.Join(home, ".litertlm", "models"), nil
	}
	return filepath.Join(base, "litertlm-go", "models"), nil
}
