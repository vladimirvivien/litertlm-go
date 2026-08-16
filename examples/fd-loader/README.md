# File Descriptor Loader (`examples/fd-loader`)

This example demonstrates how to initialize the LiteRT-LM engine using an open operating system file descriptor (`*os.File`) instead of a filesystem path string via `litertlm.WithModelFd`.

## How It Works

1. Opens the target `.litertlm` model file with standard `os.Open`.
2. Passes `litertlm.WithModelFd(f)` to `litertlm.New`.
3. LiteRT-LM mmaps the model directly from the provided descriptor.

> **Note**: File descriptor loading is supported on Linux and macOS. On Windows, CRT file descriptor boundary constraints require file paths.

## Usage

```bash
go run . -model /path/to/model.litertlm -prompt "What is the capital of France?"
```

With automatic library and model provisioning:
```bash
go run . -get-lib v0.16.0 -get-model litert-community/gemma3-1b-it-int4
```
