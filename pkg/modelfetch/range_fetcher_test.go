package modelfetch

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestHTTPRangeFetcher_RangesAndOptions(t *testing.T) {
	payload := []byte("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
	modTime := time.Date(2026, 11, 20, 0, 0, 0, 0, time.UTC)

	var seenAuth string
	var seenRange string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenAuth = r.Header.Get("Authorization")
		seenRange = r.Header.Get("Range")
		http.ServeContent(w, r, "model.litertlm", modTime, bytes.NewReader(payload))
	}))
	defer srv.Close()

	fetcher := NewHTTPRangeFetcher(srv.Client(), srv.URL+"/model.litertlm", WithAuthToken("tok-123"))

	targetURL, err := fetcher.TargetURL()
	if err != nil || targetURL != srv.URL+"/model.litertlm" {
		t.Fatalf("TargetURL() = (%q, %v)", targetURL, err)
	}

	// Offset 0 slice
	var buf bytes.Buffer
	if err = fetcher.FetchRange(context.Background(), 0, 10, &buf); err != nil {
		t.Fatalf("FetchRange(0, 10): %v", err)
	}
	if buf.String() != string(payload[:10]) {
		t.Errorf("got %q, want %q", buf.String(), payload[:10])
	}
	if seenAuth != "Bearer tok-123" || seenRange != "bytes=0-9" {
		t.Errorf("headers auth=%q range=%q", seenAuth, seenRange)
	}

	// Mid-file slice
	buf.Reset()
	if err = fetcher.FetchRange(context.Background(), 10, 16, &buf); err != nil {
		t.Fatalf("FetchRange(10, 16): %v", err)
	}
	if buf.String() != string(payload[10:26]) {
		t.Errorf("got %q, want %q", buf.String(), payload[10:26])
	}

	// End-of-file slice
	buf.Reset()
	endOff := uint64(len(payload) - 6)
	if err = fetcher.FetchRange(context.Background(), endOff, 6, &buf); err != nil {
		t.Fatalf("FetchRange(end, 6): %v", err)
	}
	if buf.String() != string(payload[endOff:]) {
		t.Errorf("got %q, want %q", buf.String(), payload[endOff:])
	}

	// Zero-length fetch is a no-op
	buf.Reset()
	if err = fetcher.FetchRange(context.Background(), 12, 0, &buf); err != nil {
		t.Fatalf("FetchRange zero length: %v", err)
	}
	if buf.Len() != 0 {
		t.Errorf("expected empty buffer for zero-length fetch, got %d bytes", buf.Len())
	}
}

func TestHTTPRangeFetcher_ShorthandResolution(t *testing.T) {
	f := NewHTTPRangeFetcher(nil, "litert-community/gemma-4-E2B-it-litert-lm", WithVariant("gpu"))
	got, err := f.TargetURL()
	if err != nil {
		t.Fatalf("TargetURL: %v", err)
	}
	want := "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-gpu.litertlm"
	if got != want {
		t.Errorf("TargetURL = %q, want %q", got, want)
	}
}

func TestHTTPRangeFetcher_ErrorsAndEdgeCases(t *testing.T) {
	t.Run("http_status_errors", func(t *testing.T) {
		for _, code := range []int{
			http.StatusUnauthorized,
			http.StatusForbidden,
			http.StatusNotFound,
			http.StatusRequestedRangeNotSatisfiable,
			http.StatusTooManyRequests,
			http.StatusServiceUnavailable,
		} {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(code)
			}))
			f := NewHTTPRangeFetcher(srv.Client(), srv.URL+"/m.litertlm")
			var buf bytes.Buffer
			err := f.FetchRange(context.Background(), 0, 8, &buf)
			srv.Close()

			var rfe *RangeFetchError
			if !errors.As(err, &rfe) || rfe.HTTPStatusCode() != code {
				t.Errorf("status %d -> err = %v, want *RangeFetchError(%d)", code, err, code)
			}
		}
	})

	t.Run("200_ok_without_range_support", func(t *testing.T) {
		full := []byte("0123456789ABCDEF")
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Length", strconv.Itoa(len(full)))
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(full)
		}))
		defer srv.Close()

		f := NewHTTPRangeFetcher(srv.Client(), srv.URL+"/m.litertlm")
		var buf bytes.Buffer

		// Nonzero offset must reject 200 OK
		if err := f.FetchRange(context.Background(), 4, 4, &buf); err == nil {
			t.Error("expected error on 200 OK with nonzero offset")
		}

		// Offset 0 with exact full size succeeds cleanly
		buf.Reset()
		if err := f.FetchRange(context.Background(), 0, uint64(len(full)), &buf); err != nil {
			t.Fatalf("offset 0 exact full size failed: %v", err)
		}
		if buf.String() != string(full) {
			t.Errorf("got %q, want %q", buf.String(), full)
		}

		// Offset 0 requesting slice smaller than 200 OK body fails cleanly
		buf.Reset()
		if err := f.FetchRange(context.Background(), 0, 4, &buf); err == nil {
			t.Error("expected error when 200 OK Content-Length exceeds requested length")
		}
	})

	t.Run("short_partial_body", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusPartialContent)
			_, _ = w.Write([]byte("abc"))
		}))
		defer srv.Close()

		f := NewHTTPRangeFetcher(srv.Client(), srv.URL+"/m.litertlm")
		var buf bytes.Buffer
		err := f.FetchRange(context.Background(), 0, 10, &buf)
		if !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Errorf("expected io.ErrUnexpectedEOF, got %v", err)
		}
	})

	t.Run("validation_overflow_and_cancel", func(t *testing.T) {
		var nilFetcher *HTTPRangeFetcher
		var buf bytes.Buffer
		if err := nilFetcher.FetchRange(context.Background(), 0, 4, &buf); err == nil {
			t.Error("expected error on nil fetcher")
		}
		if _, err := nilFetcher.TargetURL(); err == nil {
			t.Error("expected error on nil TargetURL")
		}

		f := NewHTTPRangeFetcher(nil, "https://example.com/model.litertlm", WithTimeout(time.Second))
		if err := f.FetchRange(context.Background(), 0, 4, nil); err == nil {
			t.Error("expected error on nil writer")
		}
		if err := f.FetchRange(context.Background(), math.MaxUint64-2, 8, &buf); err == nil || !strings.Contains(err.Error(), "overflow") {
			t.Errorf("expected uint64 overflow error, got %v", err)
		}

		canceled, cancel := context.WithCancel(context.Background())
		cancel()
		if err := f.FetchRange(canceled, 0, 4, &buf); !errors.Is(err, context.Canceled) {
			t.Errorf("expected context.Canceled, got %v", err)
		}
		if msg := (&RangeFetchError{StatusCode: 500, URL: "https://example.com"}).Error(); !strings.Contains(msg, "500") {
			t.Errorf("unexpected RangeFetchError.Error() = %q", msg)
		}
		if (*RangeFetchError)(nil).Error() != "" || (*RangeFetchError)(nil).HTTPStatusCode() != 0 {
			t.Error("expected zero values from nil *RangeFetchError")
		}
	})
}
