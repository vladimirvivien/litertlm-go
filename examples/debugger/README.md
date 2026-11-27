# AI Edge Debugger Captures

This example shows how to configure the AI Edge Debugger (`litertlm.WithDebuggerEnabled(true)` paired with a writable `litertlm.WithCacheDir`) and inspect trace output locations via `chat.DebugCaptureDir()` and `session.DebugCaptureDir()`.

> [!WARNING]
> Debugger traces record unencrypted prompt/decoded text and intermediate activation tensors (Safetensors and JSONL logs) under `<cache_dir>/litert_lm_debugger/<session_id>/`. Enable only for local diagnostics and delete capture files when finished.

Enabling the debugger via `WithDebuggerEnabled(true)` requires a LiteRT-LM C API `1.1.0+` (`v0.20.0`) shared library built with debugger support. On C API `1.0.0` (`v0.19.1`), the example falls back cleanly and shows `DebugCaptureDir()` returning an empty string when inactive.

## Running

```bash
go run ./examples/debugger \
  -get-lib v0.19.1 \
  -get-model litert-community/gemma3-1b-it-int4 \
  -cache-dir /tmp/litertlm-debug-cache
```

Enable debugger tracing on a `v0.20.0` (`1.1.0+`) library:

```bash
LITERTLM_LIB=/abs/path/to/v0.20.0/lib \
LITERTLM_MODEL=/abs/path/to/model.litertlm \
go run ./examples/debugger -enable-debugger -cache-dir /tmp/litertlm-debug-cache
```
