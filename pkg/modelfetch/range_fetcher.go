package modelfetch

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
)

// RangeFetchError reports a non-success HTTP response returned while fetching a
// byte range from a remote model artifact.
type RangeFetchError struct {
	StatusCode int
	URL        string
	Message    string
}

// Error formats the HTTP range failure with status code and target URL.
func (e *RangeFetchError) Error() string {
	if e == nil {
		return ""
	}
	if e.Message != "" {
		return fmt.Sprintf("modelfetch: range fetch HTTP %d (%s): %s", e.StatusCode, e.Message, e.URL)
	}
	return fmt.Sprintf("modelfetch: range fetch HTTP %d (%s)", e.StatusCode, e.URL)
}

// HTTPStatusCode returns the HTTP response status code associated with the error.
func (e *RangeFetchError) HTTPStatusCode() int {
	if e == nil {
		return 0
	}
	return e.StatusCode
}

// HTTPRangeFetcher fetches exact HTTP byte ranges for differential model sync.
type HTTPRangeFetcher struct {
	modelIDOrURL string
	cfg          config
}

// NewHTTPRangeFetcher returns an HTTPRangeFetcher configured with client (or the
// default HTTP client when nil), a direct URL or Hugging Face model shorthand,
// and standard modelfetch options (WithAuthToken, WithVariant, WithTimeout).
func NewHTTPRangeFetcher(client *http.Client, modelIDOrURL string, opts ...Option) *HTTPRangeFetcher {
	c := config{
		httpClient: client,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&c)
		}
	}
	return &HTTPRangeFetcher{
		modelIDOrURL: modelIDOrURL,
		cfg:          c,
	}
}

// TargetURL resolves the backing download URL from the configured model ID or URL.
func (f *HTTPRangeFetcher) TargetURL() (string, error) {
	if f == nil {
		return "", fmt.Errorf("modelfetch: HTTPRangeFetcher is nil")
	}
	target, err := ResolveModelIdentifierWithVariant(f.modelIDOrURL, f.cfg.variant)
	if err != nil {
		return "", err
	}
	return target.URL, nil
}

// FetchRange downloads the byte range [offset, offset+length) and streams it to w.
func (f *HTTPRangeFetcher) FetchRange(ctx context.Context, offset, length uint64, w io.Writer) error {
	if f == nil {
		return fmt.Errorf("modelfetch: HTTPRangeFetcher is nil")
	}
	if w == nil {
		return fmt.Errorf("modelfetch: range writer is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if length == 0 {
		return nil
	}
	if length > math.MaxInt64 {
		return fmt.Errorf("modelfetch: range length %d exceeds int64 limit", length)
	}
	if offset > math.MaxUint64-(length-1) {
		return fmt.Errorf("modelfetch: range [%d, +%d) overflows uint64", offset, length)
	}

	targetURL, err := f.TargetURL()
	if err != nil {
		return err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, targetURL, nil)
	if err != nil {
		return fmt.Errorf("modelfetch: create range request: %w", err)
	}
	authToken := cmp.Or(f.cfg.authToken, os.Getenv("HF_TOKEN"), os.Getenv("HUGGING_FACE_HUB_TOKEN"))
	if authToken != "" {
		req.Header.Set("Authorization", "Bearer "+authToken)
	}
	end := offset + length - 1
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", offset, end))

	client := cmp.Or(f.cfg.httpClient, defaultHTTPClient)
	if f.cfg.timeout > 0 {
		client = &http.Client{
			Timeout:       f.cfg.timeout,
			Transport:     client.Transport,
			CheckRedirect: client.CheckRedirect,
			Jar:           client.Jar,
		}
	}

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("modelfetch: range request failed: %w", err)
	}
	defer func() {
		if resp.Body != nil {
			_ = resp.Body.Close()
		}
	}()

	switch resp.StatusCode {
	case http.StatusPartialContent:
		return copyExactRange(w, resp.Body, int64(length))

	case http.StatusOK:
		// Only accept 200 OK if the request starts at 0 and matches the requested byte count.
		if offset != 0 || (resp.ContentLength >= 0 && resp.ContentLength != int64(length)) {
			return &RangeFetchError{
				StatusCode: resp.StatusCode,
				URL:        targetURL,
				Message:    "server does not support HTTP Range requests",
			}
		}
		if err = copyExactRange(w, resp.Body, int64(length)); err != nil {
			return err
		}
		var trailing [1]byte
		n, _ := resp.Body.Read(trailing[:])
		if n > 0 {
			return &RangeFetchError{
				StatusCode: resp.StatusCode,
				URL:        targetURL,
				Message:    "server returned more bytes than requested range",
			}
		}
		return nil

	default:
		return &RangeFetchError{
			StatusCode: resp.StatusCode,
			URL:        targetURL,
			Message:    resp.Status,
		}
	}
}

func copyExactRange(w io.Writer, r io.Reader, n int64) error {
	written, err := io.CopyN(w, r, n)
	if err != nil {
		if errors.Is(err, io.EOF) && written < n {
			return fmt.Errorf("modelfetch: short range body (%d/%d bytes): %w", written, n, io.ErrUnexpectedEOF)
		}
		return fmt.Errorf("modelfetch: copy range body: %w", err)
	}
	return nil
}
