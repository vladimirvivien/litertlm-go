# Example: Speculative Decoding Comparison

Compares decode throughput across `SpeculativeDecodingOff`,
`SpeculativeDecodingOn`, and `SpeculativeDecodingAdaptive` using
`litertlm.WithSpeculativeDecoding` and optional acceptance `-lenience`.

Speculative (multi-token-prediction) decoding lets the model emit several
candidate tokens per forward pass and accept whichever agree with the
verification step. `SpeculativeDecodingAdaptive` automatically disables
speculation when draft verification does not outpace plain decoding.

## What this example shows

- Configuring `litertlm.SpeculativeDecodingConfig{Mode, Lenience}` via
  `litertlm.WithSpeculativeDecoding`.
- Capturing per-session benchmarks including `MaxNumTokens` alongside
  prefill and decode throughput.
- Comparing decode throughput across `Off`, `On`, and `Adaptive`.

## Prerequisites

1. LiteRT-LM shared library staged in `LITERTLM_LIB` (or pass `-get-lib v0.19.1`).
2. A `.litertlm` model with a speculative decoding drafter (such as Gemma 4 E4B).

## Run

```bash
go run ./examples/speculative \
    -get-lib v0.19.1 \
    -model /abs/path/to/gemma-4-E4B-it.litertlm \
    -lenience 0.0
```

| Flag | Default | Notes |
|---|---|---|
| `-model` | (required) | Path to a `.litertlm` model file (falls back to `LITERTLM_MODEL`) |
| `-get-model` | unset | Optional Hugging Face repo or URL |
| `-get-lib` | unset | Optional library version to stage (e.g. `v0.19.1`) |
| `-lib` | `$LITERTLM_LIB` | Directory holding `liblitert-lm` |
| `-backend` | `"cpu"` | Inference backend (`cpu` \| `gpu`) |
| `-lenience` | `0` | Acceptance lenience in `[0, 1]` (`0` = lossless) |
| `-max-output` | `128` | Decode token limit per call |
