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
	return ResolveModelIdentifierWithVariant(raw, "")
}

// ResolveModelIdentifierWithVariant resolves a model identifier with an optional hardware/runtime variant
// (e.g. "gpu", "cpu", "web").
func ResolveModelIdentifierWithVariant(raw string, variant string) (ResolvedTarget, error) {
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

		if strings.ToLower(variant) == "gpu" && !strings.Contains(filename, "-gpu") && strings.HasSuffix(filename, ".litertlm") {
			filename = strings.TrimSuffix(filename, ".litertlm") + "-gpu.litertlm"
			dir := filepath.ToSlash(filepath.Dir(parsed.Path))
			if dir == "." || dir == "/" {
				parsed.Path = "/" + filename
			} else {
				parsed.Path = dir + "/" + filename
			}
		}

		return ResolvedTarget{
			URL:      parsed.String(),
			Filename: filename,
		}, nil
	}

	// 2. Strip optional "hf:" prefix
	clean := strings.TrimPrefix(raw, "hf:")

	// 3. Check for colon-delimited format: repo:filename or repo:variant (e.g. repo:gpu or repo:model.litertlm)
	if parts := strings.Split(clean, ":"); len(parts) == 2 {
		repoPart := strings.Trim(parts[0], "/")
		spec := strings.TrimSpace(parts[1])
		if repoPart != "" && spec != "" {
			// If spec is a variant keyword like "gpu", "cpu", "web"
			switch strings.ToLower(spec) {
			case "gpu", "cpu", "web":
				return resolveRepoWithVariant(repoPart, strings.ToLower(spec))
			default:
				// Specific filename
				filename := spec
				if !strings.HasSuffix(strings.ToLower(filename), ".litertlm") {
					filename += ".litertlm"
				}
				repo := repoPart
				if !strings.Contains(repo, "/") {
					repo = "litert-community/" + repo
				}
				return ResolvedTarget{
					URL:      fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repo, filename),
					Filename: filename,
				}, nil
			}
		}
	}

	// 4. Check for format: repo_owner/repo_name/filename.litertlm (3+ slash parts ending in .litertlm)
	slashParts := strings.Split(clean, "/")
	if len(slashParts) >= 3 && strings.HasSuffix(strings.ToLower(slashParts[len(slashParts)-1]), ".litertlm") {
		filename := slashParts[len(slashParts)-1]
		repo := strings.Join(slashParts[:len(slashParts)-1], "/")
		return ResolvedTarget{
			URL:      fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repo, filename),
			Filename: filename,
		}, nil
	}

	// 5. Check for 2-part repo: repo_owner/model_or_repo
	if len(slashParts) == 2 {
		return resolveOwnerRepo(slashParts[0], slashParts[1], variant)
	}

	// 6. 1-part shorthand: e.g. "gemma-4-E4B-it", "gemma-4-E4B-it-gpu", or "gemma3-1b-it-int4"
	if len(slashParts) == 1 {
		return resolveShorthand(slashParts[0], variant)
	}

	return ResolvedTarget{}, fmt.Errorf("modelfetch: unparseable model identifier %q", raw)
}

func resolveRepoWithVariant(repo string, variant string) (ResolvedTarget, error) {
	slashParts := strings.Split(repo, "/")
	if len(slashParts) == 2 {
		return resolveOwnerRepo(slashParts[0], slashParts[1], variant)
	}
	return resolveShorthand(repo, variant)
}

func resolveOwnerRepo(owner, rawModel string, variant string) (ResolvedTarget, error) {
	rawModel = strings.TrimSuffix(rawModel, ".litertlm")

	// Detect embedded variant in model name if not overridden (e.g. gemma-4-E4B-it-gpu)
	effectiveVariant := variant
	if effectiveVariant == "" {
		if strings.HasSuffix(strings.ToLower(rawModel), "-gpu") {
			effectiveVariant = "gpu"
			rawModel = strings.TrimSuffix(rawModel, "-gpu")
			rawModel = strings.TrimSuffix(rawModel, "-GPU")
		} else if strings.HasSuffix(strings.ToLower(rawModel), "-cpu") {
			effectiveVariant = "cpu"
			rawModel = strings.TrimSuffix(rawModel, "-cpu")
			rawModel = strings.TrimSuffix(rawModel, "-CPU")
		}
	}

	repo := rawModel
	var baseFile string

	if owner == "litert-community" {
		// In litert-community:
		// Repos ending in -litert-lm (e.g. gemma-4-E4B-it-litert-lm) contain files named gemma-4-E4B-it.litertlm
		if strings.HasSuffix(rawModel, "-litert-lm") {
			repo = rawModel
			baseFile = strings.TrimSuffix(rawModel, "-litert-lm")
		} else {
			// Shorthand under litert-community like gemma-4-E4B-it
			baseFile = rawModel
			if strings.HasPrefix(strings.ToLower(rawModel), "gemma-4-") || strings.HasPrefix(strings.ToLower(rawModel), "gemma4-") {
				repo = rawModel + "-litert-lm"
			} else {
				repo = rawModel
			}
		}
	} else {
		baseFile = rawModel
	}

	filename := baseFile
	if strings.ToLower(effectiveVariant) == "gpu" && !strings.HasSuffix(filename, "-gpu") {
		filename += "-gpu"
	}
	if strings.ToLower(effectiveVariant) == "web" && !strings.HasSuffix(filename, "-web") {
		filename += "-web"
	}
	filename += ".litertlm"

	return ResolvedTarget{
		URL:      fmt.Sprintf("https://huggingface.co/%s/%s/resolve/main/%s", owner, repo, filename),
		Filename: filename,
	}, nil
}

func resolveShorthand(rawModel string, variant string) (ResolvedTarget, error) {
	return resolveOwnerRepo("litert-community", rawModel, variant)
}
