# Centralized Config File (`examples/config-file`)

This example demonstrates how to configure a `litertlm.Client` from a centralized `config.json` configuration file using `litertlm.WithConfigFile`.

## How It Works

1. Reads backend, token limits, and sampler parameters from a JSON configuration file containing default and model-specific profiles.
2. Applies `litertlm.WithConfigFile(path, modelID)` to set initial configuration defaults.
3. Explicit options passed after `WithConfigFile` override the file settings.

## Example `config.json`

```json
{
  "default": {
    "backend": "cpu",
    "max_tokens": 1024,
    "temperature": 0.7,
    "top_k": 40
  },
  "gemma": {
    "backend": "cpu",
    "max_tokens": 2048,
    "temperature": 0.2,
    "top_k": 20
  }
}
```

## Usage

```bash
go run . -model /path/to/model.litertlm -config config.json -profile gemma
```

With automatic library and model provisioning:
```bash
go run . -get-lib v0.16.0 -get-model litert-community/gemma3-1b-it-int4 -config config.json
```
