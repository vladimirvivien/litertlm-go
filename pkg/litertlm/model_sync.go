package litertlm

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/vladimirvivien/litertlm-go/internal/cstr"
)

// RangeFetcher downloads a byte range [offset, offset+length) into w.
// Implementations include *modelfetch.HTTPRangeFetcher.
type RangeFetcher interface {
	FetchRange(ctx context.Context, offset, length uint64, w io.Writer) error
}

// RangeFetcherFunc adapts an ordinary function to RangeFetcher.
type RangeFetcherFunc func(ctx context.Context, offset, length uint64, w io.Writer) error

// FetchRange calls f(ctx, offset, length, w).
func (f RangeFetcherFunc) FetchRange(ctx context.Context, offset, length uint64, w io.Writer) error {
	return f(ctx, offset, length, w)
}

// ModelSyncPlan summarizes the byte-level diff calculated between a local
// .litertlm file and a remote source.
type ModelSyncPlan struct {
	UpToDate        bool
	BytesToDownload uint64
	BytesToReuse    uint64
	TotalRemoteSize uint64
	Summary         string
}

// ModelSyncOption configures PlanModelSync, SyncModel, and SyncModelCache.
type ModelSyncOption func(*modelSyncConfig)

type modelSyncConfig struct {
	libPath               string
	fullDownloadThreshold *float64
	clean                 *bool
	checkDiskSpace        *bool
	allowOfflineFallback  *bool
	err                   error
}

// WithFullDownloadThreshold sets the changed-byte ratio in [0.0, 1.0] above
// which differential sync falls back to a full download (C default: 0.80;
// 0.0 disables threshold fallback).
func WithFullDownloadThreshold(threshold float64) ModelSyncOption {
	return func(c *modelSyncConfig) {
		if math.IsNaN(threshold) || threshold < 0 || threshold > 1 {
			c.err = fmt.Errorf("litertlm: invalid full-download threshold %v (want [0.0, 1.0])", threshold)
			return
		}
		c.fullDownloadThreshold = new(threshold)
	}
}

// WithCleanSync configures whether the existing local model is deleted before
// downloading when disk space is constrained.
func WithCleanSync(clean bool) ModelSyncOption {
	return func(c *modelSyncConfig) { c.clean = new(clean) }
}

// WithCheckDiskSpace toggles pre-flight free disk space verification before
// staging downloads.
func WithCheckDiskSpace(check bool) ModelSyncOption {
	return func(c *modelSyncConfig) { c.checkDiskSpace = new(check) }
}

// WithOfflineFallback configures whether SyncModelCache returns an existing
// valid cached model when remote fetches fail with ErrUnavailable or
// ErrDeadlineExceeded.
func WithOfflineFallback(allow bool) ModelSyncOption {
	return func(c *modelSyncConfig) { c.allowOfflineFallback = new(allow) }
}

// WithModelSyncLib overrides the native library directory used when PlanModelSync,
// SyncModel, or SyncModelCache initializes the runtime.
func WithModelSyncLib(dir string) ModelSyncOption {
	return func(c *modelSyncConfig) { c.libPath = dir }
}

type httpStatusCoder interface {
	error
	HTTPStatusCode() int
}

type modelSyncCallbackState struct {
	ctx     context.Context
	fetcher RangeFetcher
}

var (
	modelSyncOnce       sync.Once
	modelSyncTrampAddr  uintptr
	modelSyncRegistry   callbackRegistry[*modelSyncCallbackState]
	modelSyncWriteCheck = callRangeWriterWrite
	modelSyncErrCheck   = callRangeWriterSetError
)

// modelSyncTrampoline matches LiteRtLmModelSyncRangeFetcherCallback:
// LiteRtLmStatusCode (*)(uint64_t offset, uint64_t length, LiteRtLmModelSyncRangeWriter* writer, void* user_data)
func modelSyncTrampoline(offset, length, writer, userData uintptr) uintptr {
	state, ok := modelSyncRegistry.Lookup(userData)
	if !ok || state == nil || state.fetcher == nil || writer == 0 {
		return uintptr(StatusInvalidArgument)
	}
	w := ModelSyncRangeWriter(writer)
	err := state.fetcher.FetchRange(state.ctx, uint64(offset), uint64(length), w)
	if err == nil {
		return uintptr(StatusOk)
	}
	code := mapRangeFetchError(err)
	_ = w.SetError(err.Error())
	return uintptr(code)
}

func ensureModelSyncTrampoline() {
	modelSyncOnce.Do(func() {
		modelSyncTrampAddr = purego.NewCallback(modelSyncTrampoline)
	})
}

func mapRangeFetchError(err error) StatusCode {
	if err == nil {
		return StatusOk
	}
	if errors.Is(err, context.Canceled) {
		return StatusCancelled
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return StatusDeadlineExceeded
	}
	if ce, ok := errors.AsType[*CError](err); ok && ce.Code != StatusOk {
		return ce.Code
	}
	if hs, ok := errors.AsType[httpStatusCoder](err); ok {
		switch sc := hs.HTTPStatusCode(); {
		case sc == http.StatusUnauthorized:
			return StatusUnauthenticated
		case sc == http.StatusForbidden:
			return StatusPermissionDenied
		case sc == http.StatusNotFound:
			return StatusNotFound
		case sc == http.StatusRequestedRangeNotSatisfiable:
			return StatusOutOfRange
		case sc == http.StatusTooManyRequests || sc >= 500:
			return StatusUnavailable
		case sc >= 400 && sc < 500:
			return StatusInvalidArgument
		}
	}
	if ne, ok := errors.AsType[net.Error](err); ok {
		if ne.Timeout() {
			return StatusDeadlineExceeded
		}
		return StatusUnavailable
	}
	for _, entry := range []struct {
		sentinel error
		code     StatusCode
	}{
		{ErrUnavailable, StatusUnavailable},
		{ErrDeadlineExceeded, StatusDeadlineExceeded},
		{ErrCancelled, StatusCancelled},
		{ErrNotFound, StatusNotFound},
		{ErrAlreadyExists, StatusAlreadyExists},
		{ErrPermissionDenied, StatusPermissionDenied},
		{ErrUnauthenticated, StatusUnauthenticated},
		{ErrOutOfRange, StatusOutOfRange},
		{ErrDataLoss, StatusDataLoss},
		{ErrResourceExhausted, StatusResourceExhausted},
		{ErrFailedPrecondition, StatusFailedPrecondition},
		{ErrAborted, StatusAborted},
		{ErrInvalidArgument, StatusInvalidArgument},
		{ErrUnimplemented, StatusUnimplemented},
		{ErrUnknown, StatusUnknown},
	} {
		if errors.Is(err, entry.sentinel) {
			return entry.code
		}
	}
	return StatusInternal
}

var callRangeWriterWrite = func(w ModelSyncRangeWriter, dataPtr unsafe.Pointer, size uint64) error {
	var status int32
	if err := modelSyncRangeWriterWriteFunc.callErr(
		unsafe.Pointer(&status),
		unsafe.Pointer(&w),
		unsafe.Pointer(&dataPtr),
		unsafe.Pointer(&size),
	); err != nil {
		return err
	}
	if code := StatusCode(status); code != StatusOk {
		return &CError{Op: "model_sync_range_writer_write", Code: code}
	}
	return nil
}

var callRangeWriterSetError = func(w ModelSyncRangeWriter, msgPtr *byte) error {
	var status int32
	if err := modelSyncRangeWriterSetErrorFunc.callErr(
		unsafe.Pointer(&status),
		unsafe.Pointer(&w),
		unsafe.Pointer(&msgPtr),
	); err != nil {
		return err
	}
	if code := StatusCode(status); code != StatusOk {
		return &CError{Op: "model_sync_range_writer_set_error", Code: code}
	}
	return nil
}

// Write streams fetched bytes into the C ModelSync writer handle.
func (w ModelSyncRangeWriter) Write(p []byte) (int, error) {
	if w == 0 {
		return 0, fmt.Errorf("litertlm: model_sync_range_writer_write: invalid writer")
	}
	if len(p) == 0 {
		return 0, nil
	}
	dataPtr := unsafe.Pointer(&p[0])
	if err := modelSyncWriteCheck(w, dataPtr, uint64(len(p))); err != nil {
		return 0, err
	}
	runtime.KeepAlive(p)
	return len(p), nil
}

// SetError records a diagnostic message on w before returning a failure status
// from a RangeFetcher callback. Passing an empty string clears any prior error.
func (w ModelSyncRangeWriter) SetError(msg string) error {
	if w == 0 {
		return fmt.Errorf("litertlm: model_sync_range_writer_set_error: invalid writer")
	}
	ptr, err := bytePtrOrNil(msg)
	if err != nil {
		return err
	}
	err = modelSyncErrCheck(w, ptr)
	runtime.KeepAlive(ptr)
	return err
}

// ---- ModelSyncOptionsHandle ----

// NewModelSyncOptions allocates a ModelSync options handle initialized to C
// defaults (requires LiteRT-LM C API 1.1.0+). Caller must Delete().
func NewModelSyncOptions() (ModelSyncOptionsHandle, error) {
	if err := requireAPIVersion(1, 1); err != nil {
		return 0, err
	}
	return callStatusHandle[ModelSyncOptionsHandle](modelSyncOptionsCreateFunc, "model_sync_options_create")
}

// Delete releases a ModelSyncOptionsHandle.
func (o ModelSyncOptionsHandle) Delete() { deleteHandle(o, modelSyncOptionsDeleteFunc) }

// SetFullDownloadThreshold sets the change ratio in [0.0, 1.0] above which
// sync falls back to downloading the complete remote file.
func (o ModelSyncOptionsHandle) SetFullDownloadThreshold(threshold float64) error {
	if o == 0 {
		return fmt.Errorf("litertlm: model_sync_options_set_full_download_threshold: invalid options")
	}
	if math.IsNaN(threshold) || threshold < 0 || threshold > 1 {
		return fmt.Errorf("litertlm: model_sync_options_set_full_download_threshold: invalid threshold %v (want [0.0, 1.0])", threshold)
	}
	return setHandleScalar(o, "model sync options", modelSyncOptionsSetFullDownloadThresholdFunc,
		"model_sync_options_set_full_download_threshold", threshold)
}

// SetClean toggles removing the local model upfront when disk space is limited.
func (o ModelSyncOptionsHandle) SetClean(clean bool) error {
	return setHandleBool(o, "model sync options", modelSyncOptionsSetCleanFunc,
		"model_sync_options_set_clean", clean)
}

// SetCheckDiskSpace toggles pre-flight available disk space verification.
func (o ModelSyncOptionsHandle) SetCheckDiskSpace(check bool) error {
	return setHandleBool(o, "model sync options", modelSyncOptionsSetCheckDiskSpaceFunc,
		"model_sync_options_set_check_disk_space", check)
}

// SetAllowOfflineFallback toggles falling back to a cached model on network failure.
func (o ModelSyncOptionsHandle) SetAllowOfflineFallback(allow bool) error {
	return setHandleBool(o, "model sync options", modelSyncOptionsSetAllowOfflineFallbackFunc,
		"model_sync_options_set_allow_offline_fallback", allow)
}

// ---- ModelSyncPlanHandle ----

// NewModelSyncPlanFromFile computes a differential sync plan comparing localPath
// (which may be empty or absent) against fetcher (requires LiteRT-LM C API 1.1.0+).
// Pass 0 for opts to use defaults. Caller must Delete() the returned handle.
func NewModelSyncPlanFromFile(ctx context.Context, localPath string, fetcher RangeFetcher, opts ModelSyncOptionsHandle) (ModelSyncPlanHandle, error) {
	if err := validateSyncCall(ctx, fetcher); err != nil {
		return 0, err
	}
	if err := requireAPIVersion(1, 1); err != nil {
		return 0, err
	}
	localPtr, err := bytePtrOrNil(localPath)
	if err != nil {
		return 0, err
	}

	cookie, cbAddr, cleanup := registerSyncFetcher(ctx, fetcher)
	defer cleanup()

	plan, err := callStatusHandle[ModelSyncPlanHandle](
		modelSyncPlanCreateFromFileFunc,
		"model_sync_plan_create_from_file",
		unsafe.Pointer(&localPtr),
		unsafe.Pointer(&cbAddr),
		unsafe.Pointer(&cookie),
		unsafe.Pointer(&opts),
	)
	runtime.KeepAlive(localPtr)
	return plan, err
}

// Delete releases a ModelSyncPlanHandle.
func (p ModelSyncPlanHandle) Delete() { deleteHandle(p, modelSyncPlanDeleteFunc) }

// IsUpToDate reports whether the local model already matches remote sections.
func (p ModelSyncPlanHandle) IsUpToDate() (bool, error) {
	return getHandleBoolErr(p, "model sync plan", modelSyncPlanIsUpToDateFunc, "model_sync_plan_is_up_to_date")
}

// BytesToDownload returns the number of bytes that must be downloaded.
func (p ModelSyncPlanHandle) BytesToDownload() (uint64, error) {
	return getHandleScalarErr[ModelSyncPlanHandle, uint64](p, "model sync plan",
		modelSyncPlanGetBytesToDownloadFunc, "model_sync_plan_get_bytes_to_download")
}

// BytesToReuse returns the number of bytes reusable from the local file.
func (p ModelSyncPlanHandle) BytesToReuse() (uint64, error) {
	return getHandleScalarErr[ModelSyncPlanHandle, uint64](p, "model sync plan",
		modelSyncPlanGetBytesToReuseFunc, "model_sync_plan_get_bytes_to_reuse")
}

// TotalRemoteSize returns the total size in bytes of the remote model.
func (p ModelSyncPlanHandle) TotalRemoteSize() (uint64, error) {
	return getHandleScalarErr[ModelSyncPlanHandle, uint64](p, "model sync plan",
		modelSyncPlanGetTotalRemoteSizeFunc, "model_sync_plan_get_total_remote_size")
}

// Format returns a human-readable summary string copied into Go memory.
func (p ModelSyncPlanHandle) Format() (string, error) {
	return getHandleStringErr(p, "model sync plan", modelSyncPlanFormatFunc, "model_sync_plan_format")
}

// Snapshot reads all fields of the sync plan into a Go ModelSyncPlan value.
func (p ModelSyncPlanHandle) Snapshot() (*ModelSyncPlan, error) {
	if p == 0 {
		return nil, fmt.Errorf("litertlm: model_sync_plan: invalid plan")
	}
	upToDate, err := p.IsUpToDate()
	if err != nil {
		return nil, err
	}
	toDownload, err := p.BytesToDownload()
	if err != nil {
		return nil, err
	}
	toReuse, err := p.BytesToReuse()
	if err != nil {
		return nil, err
	}
	totalRemote, err := p.TotalRemoteSize()
	if err != nil {
		return nil, err
	}
	summary, err := p.Format()
	if err != nil {
		return nil, err
	}
	return &ModelSyncPlan{
		UpToDate:        upToDate,
		BytesToDownload: toDownload,
		BytesToReuse:    toReuse,
		TotalRemoteSize: totalRemote,
		Summary:         summary,
	}, nil
}

// ---- ModelSyncResultHandle & Execution Functions ----

// Delete releases a ModelSyncResultHandle.
func (r ModelSyncResultHandle) Delete() { deleteHandle(r, modelSyncResultDeleteFunc) }

// CachedPath returns the filesystem path of the synchronized cached model.
func (r ModelSyncResultHandle) CachedPath() (string, error) {
	return getHandleStringErr(r, "model sync result", modelSyncResultGetCachedPathFunc,
		"model_sync_result_get_cached_path")
}

// ModelSyncUpdate synchronizes localPath against fetcher and writes the updated
// artifact to outputPath (requires LiteRT-LM C API 1.1.0+).
func ModelSyncUpdate(ctx context.Context, localPath, outputPath string, fetcher RangeFetcher, opts ModelSyncOptionsHandle) error {
	if err := validateSyncCall(ctx, fetcher); err != nil {
		return err
	}
	if outputPath == "" {
		return fmt.Errorf("litertlm: model_sync_update: output model path required")
	}
	if err := requireAPIVersion(1, 1); err != nil {
		return err
	}
	localPtr, err := bytePtrOrNil(localPath)
	if err != nil {
		return err
	}
	outPtr, err := cstr.BytePtrFromString(outputPath)
	if err != nil {
		return err
	}

	cookie, cbAddr, cleanup := registerSyncFetcher(ctx, fetcher)
	defer cleanup()

	err = callStatus(
		modelSyncUpdateFunc,
		"model_sync_update",
		unsafe.Pointer(&localPtr),
		unsafe.Pointer(&outPtr),
		unsafe.Pointer(&cbAddr),
		unsafe.Pointer(&cookie),
		unsafe.Pointer(&opts),
	)
	runtime.KeepAlive(localPtr)
	runtime.KeepAlive(outPtr)
	return err
}

// ModelSyncCacheSync differentially updates modelID inside cacheDir (requires
// LiteRT-LM C API 1.1.0+). Caller must Delete() the returned handle.
func ModelSyncCacheSync(ctx context.Context, cacheDir, modelID string, fetcher RangeFetcher, opts ModelSyncOptionsHandle) (ModelSyncResultHandle, error) {
	if err := validateSyncCall(ctx, fetcher); err != nil {
		return 0, err
	}
	if cacheDir == "" {
		return 0, fmt.Errorf("litertlm: model_sync_cache_sync: cache dir required")
	}
	if modelID == "" {
		return 0, fmt.Errorf("litertlm: model_sync_cache_sync: model ID required")
	}
	if err := requireAPIVersion(1, 1); err != nil {
		return 0, err
	}
	dirPtr, err := cstr.BytePtrFromString(cacheDir)
	if err != nil {
		return 0, err
	}
	idPtr, err := cstr.BytePtrFromString(modelID)
	if err != nil {
		return 0, err
	}

	cookie, cbAddr, cleanup := registerSyncFetcher(ctx, fetcher)
	defer cleanup()

	res, err := callStatusHandle[ModelSyncResultHandle](
		modelSyncCacheSyncFunc,
		"model_sync_cache_sync",
		unsafe.Pointer(&dirPtr),
		unsafe.Pointer(&idPtr),
		unsafe.Pointer(&cbAddr),
		unsafe.Pointer(&cookie),
		unsafe.Pointer(&opts),
	)
	runtime.KeepAlive(dirPtr)
	runtime.KeepAlive(idPtr)
	return res, err
}

// ---- High-level Convenience APIs ----

// PlanModelSync loads the runtime if needed, computes a differential sync plan
// comparing localPath against fetcher, and returns a Go snapshot (requires
// LiteRT-LM C API 1.1.0+).
func PlanModelSync(ctx context.Context, localPath string, fetcher RangeFetcher, opts ...ModelSyncOption) (*ModelSyncPlan, error) {
	cfg, err := prepareHighLevelSync(ctx, fetcher, opts)
	if err != nil {
		return nil, err
	}
	var plan *ModelSyncPlan
	err = withModelSyncOptions(cfg, func(optHandle ModelSyncOptionsHandle) error {
		handle, callErr := NewModelSyncPlanFromFile(ctx, localPath, fetcher, optHandle)
		if callErr != nil {
			return callErr
		}
		defer handle.Delete()
		plan, callErr = handle.Snapshot()
		return callErr
	})
	return plan, err
}

// SyncModel loads the runtime if needed and differentially updates localPath into
// outputPath using fetcher (requires LiteRT-LM C API 1.1.0+). When localPath ==
// outputPath, the C runtime updates the file atomically in place.
func SyncModel(ctx context.Context, localPath, outputPath string, fetcher RangeFetcher, opts ...ModelSyncOption) error {
	if outputPath == "" {
		return fmt.Errorf("litertlm: SyncModel: output path required")
	}
	cfg, err := prepareHighLevelSync(ctx, fetcher, opts)
	if err != nil {
		return err
	}
	return withModelSyncOptions(cfg, func(optHandle ModelSyncOptionsHandle) error {
		return ModelSyncUpdate(ctx, localPath, outputPath, fetcher, optHandle)
	})
}

// SyncModelCache loads the runtime if needed, differentially synchronizes
// modelID within cacheDir using fetcher, and returns the cached model path
// (requires LiteRT-LM C API 1.1.0+).
func SyncModelCache(ctx context.Context, cacheDir, modelID string, fetcher RangeFetcher, opts ...ModelSyncOption) (string, error) {
	if cacheDir == "" {
		return "", fmt.Errorf("litertlm: SyncModelCache: cache directory required")
	}
	if modelID == "" {
		return "", fmt.Errorf("litertlm: SyncModelCache: model ID required")
	}
	cfg, err := prepareHighLevelSync(ctx, fetcher, opts)
	if err != nil {
		return "", err
	}
	var cachedPath string
	err = withModelSyncOptions(cfg, func(optHandle ModelSyncOptionsHandle) error {
		res, callErr := ModelSyncCacheSync(ctx, cacheDir, modelID, fetcher, optHandle)
		if callErr != nil {
			return callErr
		}
		defer res.Delete()
		cachedPath, callErr = res.CachedPath()
		return callErr
	})
	return cachedPath, err
}

func validateSyncCall(ctx context.Context, fetcher RangeFetcher) error {
	if fetcher == nil {
		return fmt.Errorf("litertlm: model sync: nil RangeFetcher")
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return err
		}
	}
	return nil
}

func registerSyncFetcher(ctx context.Context, fetcher RangeFetcher) (cookie, cbAddr uintptr, cleanup func()) {
	ensureModelSyncTrampoline()
	safeCtx := cmp.Or(ctx, context.Background())
	cookie = modelSyncRegistry.Register(&modelSyncCallbackState{
		ctx:     safeCtx,
		fetcher: fetcher,
	})
	return cookie, modelSyncTrampAddr, func() { modelSyncRegistry.Unregister(cookie) }
}

func prepareHighLevelSync(ctx context.Context, fetcher RangeFetcher, opts []ModelSyncOption) (modelSyncConfig, error) {
	var cfg modelSyncConfig
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	if cfg.err != nil {
		return cfg, cfg.err
	}
	if err := validateSyncCall(ctx, fetcher); err != nil {
		return cfg, err
	}
	if _, _, _, hasVer := rt.apiVersion(); !hasVer && !rt.isLoaded() {
		if err := Load(cfg.libPath, "", ""); err != nil {
			return cfg, fmt.Errorf("litertlm: model sync: %w", err)
		}
	}
	if err := requireAPIVersion(1, 1); err != nil {
		return cfg, err
	}
	return cfg, nil
}

func withModelSyncOptions(cfg modelSyncConfig, fn func(ModelSyncOptionsHandle) error) error {
	if cfg.fullDownloadThreshold == nil &&
		cfg.clean == nil &&
		cfg.checkDiskSpace == nil &&
		cfg.allowOfflineFallback == nil {
		return fn(0)
	}
	return withTempHandle(
		func() (ModelSyncOptionsHandle, error) {
			o, err := NewModelSyncOptions()
			if err != nil {
				return 0, err
			}
			if cfg.fullDownloadThreshold != nil {
				if err = o.SetFullDownloadThreshold(*cfg.fullDownloadThreshold); err != nil {
					o.Delete()
					return 0, err
				}
			}
			if cfg.clean != nil {
				if err = o.SetClean(*cfg.clean); err != nil {
					o.Delete()
					return 0, err
				}
			}
			if cfg.checkDiskSpace != nil {
				if err = o.SetCheckDiskSpace(*cfg.checkDiskSpace); err != nil {
					o.Delete()
					return 0, err
				}
			}
			if cfg.allowOfflineFallback != nil {
				if err = o.SetAllowOfflineFallback(*cfg.allowOfflineFallback); err != nil {
					o.Delete()
					return 0, err
				}
			}
			return o, nil
		},
		func(o ModelSyncOptionsHandle) { o.Delete() },
		fn,
	)
}
