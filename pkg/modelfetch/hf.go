package modelfetch

import (
	"fmt"
	"net/url"
	"path/filepath"
	"strings"
)

// ResolvedTarget holds the download URL and suggested filename.
type ResolvedTarget struct {
	URL      string
	Filename string
}

// ResolveModelIdentifier resolves a model identifier, Hugging Face repo shorthand,
// or direct URL into an HTTP download URL and filename.
func ResolveModelIdentifier(raw string) (ResolvedTarget, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ResolvedTarget{}, fmt.Errorf("modelfetch: empty model identifier")
	}

	// 1. Direct HTTP/HTTPS URL
	if strings.HasPrefix(raw, "http://") || strings.HasPrefix(raw, "https://") {
		parsed, err := url.Parse(raw)
		if err != nil {
			return ResolvedTarget{}, fmt.Errorf("modelfetch: invalid URL %q: %w", raw, err)
		}
		filename := filepath.Base(parsed.Path)
		if filename == "" || filename == "/" || filename == "." {
			filename = "model.litertlm"
		}
		return ResolvedTarget{
			URL:      raw,
			Filename: filename,
		}, nil
	}

	// 2. Strip optional "hf:" prefix
	clean := strings.TrimPrefix(raw, "hf:")

	// 3. Check for format: repo/subrepo:filename
	if parts := strings.Split(clean, ":"); len(parts) == 2 {
		repo := strings.Trim(parts[0], "/")
		filename := strings.TrimSpace(parts[1])
		if repo != "" && filename != "" {
			return ResolvedTarget{
				URL:      fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repo, filename),
				Filename: filename,
			}, nil
		}
	}

	// 4. Check for format: repo_owner/repo_name/filename.litertlm
	slashParts := strings.Split(clean, "/")
	if len(slashParts) >= 3 && strings.HasSuffix(strings.ToLower(slashParts[len(slashParts)-1]), ".litertlm") {
		filename := slashParts[len(slashParts)-1]
		repo := strings.Join(slashParts[:len(slashParts)-1], "/")
		return ResolvedTarget{
			URL:      fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repo, filename),
			Filename: filename,
		}, nil
	}

	// 5. Check for 2-part repo: repo_owner/repo_name (default filename to <repo_name>.litertlm)
	if len(slashParts) == 2 {
		repo := clean
		filename := slashParts[1]
		if !strings.HasSuffix(strings.ToLower(filename), ".litertlm") {
			filename += ".litertlm"
		}
		return ResolvedTarget{
			URL:      fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repo, filename),
			Filename: filename,
		}, nil
	}

	// 6. 1-part shorthand: assume litert-community/<name>/<name>.litertlm
	if len(slashParts) == 1 {
		name := slashParts[0]
		filename := name
		if !strings.HasSuffix(strings.ToLower(filename), ".litertlm") {
			filename += ".litertlm"
		}
		repo := fmt.Sprintf("litert-community/%s", strings.TrimSuffix(name, ".litertlm"))
		return ResolvedTarget{
			URL:      fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repo, filename),
			Filename: filename,
		}, nil
	}

	return ResolvedTarget{}, fmt.Errorf("modelfetch: unparseable model identifier %q", raw)
}
