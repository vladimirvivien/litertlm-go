# LoRA Tuner (`examples/lora-tuner`)

This example demonstrates how to configure LoRA ranks and paths on `EngineSettings` during engine initialization, as well as applying LoRA adapters dynamically to sessions.

## How It Works

1. Configures LoRA ranks on `EngineSettings` via `settings.SetLoraRanks(ranks)` or `litertlm.WithLoraPath(path)` / `litertlm.WithAudioLoraPath(path)`.
2. Initializes `litertlm.NewEngine` with the configured settings.
3. Loads LoRA weights into the active session for domain-specific fine-tuning.

## Usage

```bash
go run . -model /path/to/model.litertlm
```

With automatic library and model provisioning:
```bash
go run . -get-lib v0.16.0 -get-model litert-community/gemma3-1b-it-int4
```
