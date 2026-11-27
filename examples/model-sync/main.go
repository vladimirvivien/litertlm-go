// model-sync demonstrates differential .litertlm synchronization using
// modelfetch.NewHTTPRangeFetcher, PlanModelSync, SyncModel, and SyncModelCache
// (requires LiteRT-LM C API 1.1.0+).
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
	"github.com/vladimirvivien/litertlm-go/pkg/modelfetch"
)

func main() {
	urlOrModel := flag.String("url", "litert-community/gemma3-1b-it-int4", "remote HTTP(S) model URL or Hugging Face model ID")
	localPath := flag.String("local", "", "optional path to an existing local .litertlm file")
	cacheDir := flag.String("cache-dir", filepath.Join(os.TempDir(), "litertlm-sync-cache"), "cache directory used with -apply")
	modelID := flag.String("model-id", "gemma3-1b-it-int4", "cache model identifier used with -apply")
	threshold := flag.Float64("threshold", 0.80, "full-download fallback threshold ratio in [0.0, 1.0]")
	offlineFallback := flag.Bool("offline-fallback", true, "fall back to valid cached model on network failure")
	checkDiskSpace := flag.Bool("check-disk-space", true, "verify free disk space before staging downloads")
	clean := flag.Bool("clean", false, "delete existing local model before downloading if disk space is limited")
	applySync := flag.Bool("apply", false, "execute SyncModelCache after planning")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.20.0 or v0.19.1)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	flag.Parse()

	ctx := context.Background()
	resolvedLib := *libPath
	if *getLib != "" {
		staged, err := litertlm.FetchLib(ctx, "", "", *getLib)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fetch library: %v\n", err)
			os.Exit(1)
		}
		resolvedLib = staged
	}

	fetcher := modelfetch.NewHTTPRangeFetcher(http.DefaultClient, *urlOrModel)
	targetURL, err := fetcher.TargetURL()
	if err != nil {
		fmt.Fprintf(os.Stderr, "resolve target URL: %v\n", err)
		os.Exit(2)
	}
	fmt.Printf("Remote target: %s\n", targetURL)

	syncOpts := []litertlm.ModelSyncOption{
		litertlm.WithModelSyncLib(resolvedLib),
		litertlm.WithFullDownloadThreshold(*threshold),
		litertlm.WithOfflineFallback(*offlineFallback),
		litertlm.WithCheckDiskSpace(*checkDiskSpace),
		litertlm.WithCleanSync(*clean),
	}

	plan, err := litertlm.PlanModelSync(ctx, *localPath, fetcher, syncOpts...)
	if err != nil {
		if errors.Is(err, litertlm.ErrUnimplemented) {
			maj, minVer, pat, _ := litertlm.APIVersion()
			fmt.Printf("Differential ModelSync requires LiteRT-LM C API 1.1.0+ (v0.20.0); loaded %d.%d.%d: %v\n",
				maj, minVer, pat, err)
			return
		}
		fmt.Fprintf(os.Stderr, "PlanModelSync: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("UpToDate: %v | Download: %d B | Reuse: %d B | RemoteTotal: %d B\n%s\n",
		plan.UpToDate, plan.BytesToDownload, plan.BytesToReuse, plan.TotalRemoteSize, plan.Summary)

	if *applySync {
		cachedPath, err := litertlm.SyncModelCache(ctx, *cacheDir, *modelID, fetcher, syncOpts...)
		if err != nil {
			fmt.Fprintf(os.Stderr, "SyncModelCache: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Synchronized cached model path: %s\n", cachedPath)
	}
}
