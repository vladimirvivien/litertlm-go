# Supported Models

`litertlm-go` is compatible with any `.litertlm` model family supported by the underlying LiteRT-LM C-API runtime. This document lists the model families that have been directly tested and verified using our integration test suite.

## Tested and supported

The test (`pkg/litertlm/model_matrix_test.go`) runs three checks per
model — chat send, tokenize round-trip, streaming coherence — on the
CPU and GPU backends. Results are measured on LiteRT-LM v0.13.1. CPU
results are verified on both Linux and Windows; the GPU column is the
WebGPU / Direct3D 12 backend, measured on Windows.

| Model | Processor | CPU | GPU |
|---|---|:---:|:---:|
| gemma-4-E2B-it | `gemma4` | ✅ | ✅ |
| gemma-4-E4B-it | `gemma4` | ✅ | ✅ |
| gemma-4-12B-it | `gemma4` | ❌ | ❌ (see Caveats) |
| gemma3-1b-it-int4 | `gemma3` | ✅ | ✅ |
| gemma-3n-E2B-it-int4 | `gemma3` | ✅ | ✅ |
| gemma-3n-E4B-it-int4 | `gemma3` | ✅ | ✅ |
| Qwen3-0.6B | `qwen3` | ✅ | ✅ |
| Qwen3-4B (channelwise int8) | `qwen3` | ✅ | ✅ |
| functiongemma-270m (mobile-actions ft) | `function_gemma` | ✅ | ❌ (see Caveats) |
| Phi-4-mini-instruct | `generic` | ✅ | ✅ |

The ❌ cells are model-specific, not branch-level. gemma-4-12B fails on
both backends under v0.13.1 — a GPU-only artifact that CPU rejects, plus
an upstream GPU regression (see Caveats); FunctionGemma's GPU output is
degenerate. Every processor branch otherwise passes on both backends.

## Caveats

- **Qwen 3 multi-turn.** Single-turn text generation works. The second
  conversation turn fails inside the runtime: Qwen 3's Jinja chat
  template calls `reasoning_content.strip('\n')`, and LiteRT-LM's
  embedded Jinja lacks `string.strip(chars)`. 
- **Generic-processor families.** One-shot text
  generation works while richer chat-template and tool-call features were not
  applied.
- **FunctionGemma on GPU.** The mobile-actions q8 fine-tune loads on the
  GPU backend but emits only `<pad>` tokens; it produces correct output
  on CPU. Run it on the CPU backend.
- **gemma-4-12B fails on both backends (v0.13.1).** CPU: the published
  `.litertlm` declares a GPU-only backend constraint, so `engine_create`
  is rejected with `Main backend constraint mismatch. Model requires one
  of [gpu]`. GPU: the model fails to load under the v0.13.1 GPU runtime —
  one allocation exceeds the WebGPU/Direct3D 12 ~2 GB per-buffer limit
  (it loaded under v0.12). Tracked upstream at
  [google-ai-edge/LiteRT-LM#2461](https://github.com/google-ai-edge/LiteRT-LM/issues/2461).
  Its replies also carry thinking-format channel markers
  (e.g. `<|channel>thought`) that the wrapper does not strip.

## Running the tests

The integration battery in `pkg/litertlm/model_matrix_test.go` runs the
text base tier against every `.litertlm` file under
`LITERTLM_TEST_MODELS_DIR`. Set `LITERTLM_TEST_BACKEND=gpu` to run on
the GPU backend (default `cpu`).
