# Model Capabilities & Container Introspection

This example demonstrates how to inspect a `.litertlm` container file to query model metadata, supported input modalities, and default sampler parameters using `litertlm.InspectModel`.

Model inspection reads only FlatBuffer metadata headers. It does not allocate KV caches, load weights into memory, or compile shader pipelines.

## Prerequisites

* LiteRT-LM unified C-API shared library (`liblitert-lm.so`, `liblitert-lm.dylib`, or `litert-lm.dll`).
* A `.litertlm` model file (or use `-get-model`).

## Running the Example

### Automated Download

```bash
go run ./examples/capabilities \
  -get-lib v0.19.1 \
  -get-model litert-community/gemma3-1b-it-int4
```

### Local Paths & Environment Variables

```bash
export LITERTLM_LIB=/path/to/lib
export LITERTLM_MODEL=/path/to/model.litertlm

go run ./examples/capabilities
```

### JSON Output

Pass the `-json` flag to output structured JSON:

```bash
go run ./examples/capabilities -model /path/to/model.litertlm -json
```

## Example Output

```text
Model Capabilities for: gemma3-1b-it-int4.litertlm
--------------------------------------------------
  Model Type           : LLM
  Min Runtime Version  : 0.14.0
  Supported Modalities : [Text]
  Max Context Tokens   : 32768 (dynamic=false)
  Speculative Decoding : false
  Thinking / Reasoning : false
  Function Calling     : true
  Default Sampler      : TopK
  Default Temperature  : 1.00
  Default Top-K        : 64
  Default Top-P        : 0.95
  Vision Token Budget  : N/A
  Modality Text        : backends=[CPU]
```
