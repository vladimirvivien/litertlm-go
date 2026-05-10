# Example: Token-by-Token Streaming

Streams the assistant's reply token-by-token using `Chat.SendStream`.
The chat template is applied before each turn, so chat-tuned models
(Gemma 4, etc.) produce a proper assistant reply for any user prompt —
no special prompt shape required.

## What this example shows

- Creates a `litertlm.Client` and opens a `Chat` with a system prompt.
- Calls `chat.SendStream(ctx, prompt)` to get an `iter.Seq2[Chunk, error]`.
- Iterates with `for chunk, err := range stream`, prints `chunk.Text`
  as it arrives, and emits a final newline when `chunk.Final` is true.

For raw, non-templated streaming (e.g. against a base model or for a
completion-style prompt) see `Client.GenerateStream` in
[client.md](../../docs/client.md).

## Prerequisites

1. LiteRT-LM shared library files staged in `LITERTLM_LIB`.
2. A `.litertlm` chat-tuned model (e.g. Gemma 4).

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/stream \
    -model /abs/path/to/gemma-4-E2B-it.litertlm \
    -prompt "Write a short haiku about the sea."
```

| Flag           | Default                                   |
| -------------- | ----------------------------------------- |
| `-model`       | (required)                                |
| `-prompt`      | `"Write a short haiku about the sea."`    |
| `-system`      | `"You are a friendly assistant."`         |
| `-backend`     | `"cpu"`                                   |
| `-max`         | `1024`                                    |
| `-lib`         | `$LITERTLM_LIB`                           |
| `-speculative` | `false` — enable multi-token-prediction (MTP) speculative decoding. Requires a model with an MTP draft head bundled (e.g. `litert-community/gemma-4-E4B-it-litert-lm`). On CPU, expect a meaningful speedup only on E4B-class models; smaller models can regress. See [`examples/speculative`](../speculative) for a side-by-side benchmark. |

## Expected output

The text appears chunk-by-chunk after the `bot>` prefix, terminated by
a newline once `Final=true`:

```
user> Write a short haiku about the sea.
bot>  Blue waves crash on shore,
Salt wind whispers secrets deep,
Ocean calls to soul.
```
