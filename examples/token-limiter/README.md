# Token Limiter (`examples/token-limiter`)

This example demonstrates how to dynamically restrict token generation length on a per-call basis using `litertlm.WithMaxOutputTokens`.

## How It Works

1. Initializes a single `litertlm.Client`.
2. Generates responses using `chat.Send` with differing `litertlm.WithMaxOutputTokens` options (e.g. 10 tokens vs 100 tokens).
3. Verifies that the model truncates or stops generation within the configured token cap.

## Usage

```bash
go run . -model /path/to/model.litertlm -prompt "Write a long list of 20 historical cities."
```

With automatic library and model provisioning:
```bash
go run . -get-lib v0.16.0 -get-model litert-community/gemma3-1b-it-int4
```
