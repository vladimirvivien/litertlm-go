# Supported models

litertlm-go can load any `.litertlm` model that are supported by the underlying LiteRT-LM
runtime via a model **data processor** (see [`runtime/conversation/model_data_processor/model_data_processor_factory.cc`](https://github.com/google-ai-edge/LiteRT-LM/blob/main/runtime/conversation/model_data_processor/model_data_processor_factory.cc)). 

This document lists models that have been directly tested with `litertlm-go` with minimum version: **LiteRT-LM v0.13.1**.

## Tested and supported

The test (`pkg/litertlm/model_matrix_test.go`) runs three checks per
model — chat send, tokenize round-trip, streaming coherence — on the
CPU and GPU (WebGPU / Direct3D 12) backends.

| Model | Processor | CPU | GPU |
|---|---|---|---|
| gemma-4-E2B-it | `gemma4` | ✅ | ✅ |
| gemma-4-E4B-it | `gemma4` | ✅ | ✅ |
| gemma-4-12B-it | `gemma4` | ❌ (see Caveats) | ✅ |
| gemma3-1b-it-int4 | `gemma3` | ✅ | ✅ |
| gemma-3n-E2B-it-int4 | `gemma3` | ✅ | ✅ |
| gemma-3n-E4B-it-int4 | `gemma3` | ✅ | ✅ |
| Qwen3-0.6B | `qwen3` | ✅ | ✅ |
| Qwen3-4B (channelwise int8) | `qwen3` | ✅ | ✅ |
| functiongemma-270m (mobile-actions ft) | `function_gemma` | ✅ | ❌ (see Caveats) |
| Phi-4-mini-instruct | `generic` | ✅ | ❌ (see Caveats) |

Each ❌ is a model- or backend-specific limit, not a branch-level one:
gemma-4-12B ships GPU-only, while Phi-4-mini and FunctionGemma run on CPU
but not GPU. Every processor branch otherwise passes on both backends.
See the Caveats.

## Caveats

- **Qwen 3 multi-turn.** Single-turn text generation works. The second
  conversation turn fails inside the runtime: Qwen 3's Jinja chat
  template calls `reasoning_content.strip('\n')`, and LiteRT-LM's
  embedded Jinja lacks `string.strip(chars)`. 
- **Generic-processor families.** One-shot text
  generation works while richer chat-template and tool-call features were not
  applied.
- **Phi-4-mini on GPU.** The q8 build fails to load on the GPU (WebGPU /
  Direct3D 12) backend: one weight tensor requires ~2.46 GB, above the
  backend's ~2 GB per-allocation limit. Run it on the CPU backend.
- **FunctionGemma on GPU.** The mobile-actions q8 fine-tune loads on the
  GPU backend but emits only `<pad>` tokens; it produces correct output
  on CPU. Run it on the CPU backend.
- **gemma-4-12B on CPU.** The published `.litertlm` declares a GPU-only
  backend constraint, so `engine_create` on CPU is rejected with
  `Main backend constraint mismatch. Model requires one of [gpu]`. Run
  it on the GPU backend. Its replies also carry thinking-format channel
  markers (e.g. `<|channel>thought`) that the wrapper does not strip.

## Running the tests

The integration battery in `pkg/litertlm/model_matrix_test.go` runs the
text base tier against every `.litertlm` file under
`LITERTLM_TEST_MODELS_DIR`. Set `LITERTLM_TEST_BACKEND=gpu` to run on
the GPU backend (default `cpu`).
