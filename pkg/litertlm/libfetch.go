package litertlm

import (
	"context"
	"fmt"

	"github.com/vladimirvivien/litertlm-go/pkg/libfetch"
)

// LibFetch explicitly downloads and caches the LiteRT-LM prebuilt shared libraries
// for the given OS (e.g. "darwin", "linux", "windows"), platform/arch (e.g. "arm64", "amd64"),
// and release version (e.g. "v0.16.0").
//
// If os is empty, runtime.GOOS is used. If platform is empty, runtime.GOARCH is used.
// If version is empty, the default upstream version (v0.16.0) is used.
// Returns the local directory path containing the downloaded libraries.
func LibFetch(os, platform, version string) (string, error) {
	targetPlatform, err := libfetch.PlatformFor(os, platform)
	if err != nil {
		return "", fmt.Errorf("litertlm: LibFetch: %w", err)
	}

	if version == "" {
		version = libfetch.DefaultVersion
	}

	dir, err := libfetch.Fetch(context.Background(),
		libfetch.WithPlatform(targetPlatform),
		libfetch.WithVersion(version),
	)
	if err != nil {
		return "", fmt.Errorf("litertlm: LibFetch: %w", err)
	}

	return dir, nil
}
