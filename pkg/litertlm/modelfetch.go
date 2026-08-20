package litertlm

import (
	"context"

	"github.com/vladimirvivien/litertlm-go/pkg/modelfetch"
)

// ModelFetchOption aliases modelfetch.Option for convenience.
type ModelFetchOption = modelfetch.Option

// ModelProgressFunc aliases modelfetch.ProgressFunc.
type ModelProgressFunc = modelfetch.ProgressFunc

// ModelFetchOption aliases for convenience.
var (
	WithModelDir       = modelfetch.WithDir
	WithModelFilename  = modelfetch.WithFilename
	WithModelVariant   = modelfetch.WithVariant
	WithModelGPU       = modelfetch.WithGPU
	WithModelCPU       = modelfetch.WithCPU
	WithModelAuthToken = modelfetch.WithAuthToken
	WithModelProgress  = modelfetch.WithProgress
	WithModelTimeout   = modelfetch.WithTimeout
	WithModelSHA256    = modelfetch.WithSHA256
)

// FetchModel resolves and downloads a .litertlm model artifact from a direct URL or
// Hugging Face repository shorthand (e.g. "litert-community/gemma-4-E4B-it-litert-lm").
func FetchModel(ctx context.Context, modelIDOrURL string, opts ...ModelFetchOption) (string, error) {
	return modelfetch.Fetch(ctx, modelIDOrURL, opts...)
}
