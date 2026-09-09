package litertlm

import (
	"cmp"
	"context"
	"fmt"

	"github.com/vladimirvivien/litertlm-go/pkg/libfetch"
)

// FetchLib downloads and caches the LiteRT-LM prebuilt shared libraries
// for the given OS (e.g. "darwin", "linux", "windows"), arch (e.g. "arm64",
// "amd64"), and release version (e.g. "v0.19.1").
//
// Empty goos/goarch values default to runtime.GOOS/runtime.GOARCH; an empty
// version selects libfetch.DefaultVersion.
func FetchLib(ctx context.Context, goos, goarch, version string) (string, error) {
	targetPlatform, err := libfetch.PlatformFor(goos, goarch)
	if err != nil {
		return "", fmt.Errorf("litertlm: FetchLib: %w", err)
	}

	dir, err := libfetch.Fetch(ctx,
		libfetch.WithPlatform(targetPlatform),
		libfetch.WithVersion(cmp.Or(version, libfetch.DefaultVersion)),
	)
	if err != nil {
		return "", fmt.Errorf("litertlm: FetchLib: %w", err)
	}

	return dir, nil
}
