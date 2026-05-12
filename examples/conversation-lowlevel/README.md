# Example: Low-level Conversation API

Constructs `SessionConfig`, `ConversationConfig` and `Conversation` by
hand and runs a two-turn chat directly against the low-level
Conversation API — the same path `Client.NewChat` / `Chat.Send` build
internally.

## What this example exercises

- **Engine**: `NewEngineSettings` → `SetMaxNumTokens` →
  `EnableBenchmark` → `NewEngine`.
- **SessionConfig**: `NewSessionConfig` → `SetSamplerParams` (opt-in,
  see Notes) → `SetMaxOutputTokens` → `SetApplyPromptTemplate(true)`.
- **ConversationConfig**: `NewConversationConfig` with a JSON-encoded
  system message and empty tools/messages, then `SetExtraContext`.
- **Engine → Conversation**: `Engine.NewConversation(convCfg)`.
- **Per-turn `SendMessage`**: builds the `{role,content}` envelope
  with `encoding/json`. Reuses one `Conversation` handle for both
  turns so the KV cache persists.
- **`RenderMessage`** prints the chat template's output for a turn
  without running the model.
- **`Conversation.BenchmarkInfo()`** reports time-to-first-token and
  per-turn prefill/decode counts and throughput.

Related examples for plumbing not covered here:

- pre-seeded message history → `examples/chat-history/`
- constrained JSON output → `examples/constrained-json/`
- tool calling → `examples/conversation/`, `examples/autotool/`

## Prerequisites

1. LiteRT-LM shared library files staged in `LITERTLM_LIB`.
2. A chat-tuned `.litertlm` model (e.g. a Gemma 4 `-it` variant).

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/conversation-lowlevel \
    -model /abs/path/to/gemma-4-it.litertlm
```

Override the two turns:

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/conversation-lowlevel \
    -model /abs/path/to/gemma-4-it.litertlm \
    -turn1 "Give me three small European capital cities." \
    -turn2 "Which of those is closest to the sea?"
```

## Flags

| Flag                 | Default                                                                       | Notes                                                                                          |
| -------------------- | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `-model`             | (required)                                                                    | Path to a chat-tuned `.litertlm` model.                                                        |
| `-lib`               | `$LITERTLM_LIB`                                                               |                                                                                                |
| `-backend`           | `"cpu"`                                                                       | `cpu` or `gpu`.                                                                                |
| `-system`            | `"You are a concise assistant. Answer in one sentence."`                      | Bare content; gets JSON-encoded into `NewConversationConfig`.                                  |
| `-extra-context`     | `{"notes": [...]}`                                                            | Must be a JSON object; non-objects are dropped. Empty string disables `SetExtraContext`.       |
| `-turn1`             | `"My name is Vlad. Tell me one fun fact about the Go programming language."`  | First user message.                                                                            |
| `-turn2`             | `"What was my name again?"`                                                   | Second user message; relies on turn 1 still in the KV cache.                                   |
| `-max-tokens`        | `2048`                                                                        | `EngineSettings.SetMaxNumTokens` total budget.                                                 |
| `-max-output-tokens` | `256`                                                                         | `SessionConfig.SetMaxOutputTokens` per-turn decode cap.                                        |
| `-temp`              | `0.0`                                                                         | Sampler temperature. `0` skips `SetSamplerParams`; the C engine's default sampler is used.     |
| `-top-p`             | `0.95`                                                                        | Nucleus sampling top-p (used when `-temp > 0`).                                                |
| `-seed`              | `1`                                                                           | Sampler seed (top-p/top-k only).                                                               |

## Expected output

```
=== Rendered turn 1 (what the model actually sees) ===
<chat-template-wrapped user message — start/end markers and all>

user>      My name is Vlad. Tell me one fun fact about the Go programming language.
assistant> <one-sentence answer>

user>      What was my name again?
assistant> Your name is Vlad.

=== Conversation BenchmarkInfo ===
time to first token: 0.412s
prefill turns:       2
decode  turns:       2
  turn 0: prefill=NN tok @ NN.N tok/s | decode=NN tok @ NN.N tok/s
  turn 1: prefill=NN tok @ NN.N tok/s | decode=NN tok @ NN.N tok/s
```

## Notes

- **`SetApplyPromptTemplate(true)` is required for multi-turn chat.**
  It is what wraps each user/assistant turn in the model's start/end
  markers. With it off, the Conversation degenerates into raw
  completion.
- **Do not pass `SamplerGreedy` (type 3) to `SetSamplerParams`.** The
  C side has not implemented it yet and `engine.NewConversation` will
  fail with `UNIMPLEMENTED: Sampler type: 3 not implemented yet.`.
  Either skip `SetSamplerParams` entirely (the C engine has its own
  default) or use `SamplerTopP` / `SamplerTopK`.
- **`SetExtraContext` requires a JSON object.** Non-object input is
  silently dropped with a C-side log:
  `"Failed to parse extra context JSON or not an object"`. The chat
  template merges the object's top-level keys into the conversation
  preface.
- **System message is bare content.** Pass a JSON-encoded string
  (`"You are..."`), not a `{role,content}` envelope. The C side
  wraps it itself; passing the envelope makes the chat template drop
  the system message.
- **`SessionConfig` lifetime**: `NewConversationConfig` copies its
  contents on the C side, so the `SessionConfig` handle is safe to
  release once `ConversationConfig` exists.
- **`RenderMessage` does not call the model.** It only runs the chat
  template — no prefill, no decode.
- **`Conversation.BenchmarkInfo()` requires `EnableBenchmark` at
  engine settings time.** Without it the call still succeeds but
  every metric is zero. See `examples/benchmarks/` for the
  Response-level equivalent.
