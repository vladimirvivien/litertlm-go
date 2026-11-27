# Differential Model Sync (C API 1.1.0+)

This example demonstrates planning and applying differential `.litertlm` updates using `modelfetch.NewHTTPRangeFetcher`, `litertlm.PlanModelSync`, and `litertlm.SyncModelCache`. Unchanged sections in an existing local `.litertlm` artifact are reused while modified sections are fetched over HTTP `Range:` requests.

Differential `ModelSync` requires LiteRT-LM C API `1.1.0+` (`v0.20.0`). On C API `1.0.0` (`v0.19.1`), the program detects `ErrUnimplemented` cleanly via `errors.Is` and reports the loaded runtime version.

## Running

Plan a sync against a remote model URL or Hugging Face shorthand:

```bash
LITERTLM_LIB=/abs/path/to/v0.20.0/lib \
go run ./examples/model-sync \
  -url litert-community/gemma3-1b-it-int4 \
  -local /abs/path/to/local/gemma3-1b-it-int4.litertlm \
  -threshold 0.80
```

Apply the differential update into a local cache directory:

```bash
LITERTLM_LIB=/abs/path/to/v0.20.0/lib \
go run ./examples/model-sync \
  -url litert-community/gemma3-1b-it-int4 \
  -cache-dir /tmp/litertlm-sync-cache \
  -model-id gemma3-1b-it-int4 \
  -apply
```
