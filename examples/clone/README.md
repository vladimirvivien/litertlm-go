# clone

Demonstrates `Chat.Clone` — branching a Chat into independent
conversations that share prefilled state.

## What it does

1. Opens a Chat with a system prompt.
2. Sends one setup turn that establishes shared context ("my pet's name is Comet and he is a corgi").
3. Calls `chat.Clone()` to derive a second Chat with the same KV cache.
4. Sends a different follow-up to each branch:
   - Original: "What kind of dog is Comet?"
   - Clone:    "Suggest a fun nickname for Comet."
5. Both replies draw on the shared setup turn; the branches do not see each other's follow-ups.

## Use cases

- **Branching tool loops** — run N candidate tool calls off one prefilled prompt without re-prefilling N times.
- **Structured-output retries** — each retry starts from identical conversation state.
- **A/B decoding comparisons** — same prefix, two sampling configurations.

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
  go run ./examples/clone \
  -model /abs/path/to/gemma-4-E2B-it.litertlm
```

## Upstream support

`Chat.Clone` delegates to `Conversation.Clone`, which delegates to
the underlying executor's `Session.Clone`. As of LiteRT-LM v0.12.0
the LiteRT executor used by Gemma 4 on CPU and GPU returns
`Unimplemented`, so the example prints:

```
Chat.Clone failed: litertlm: Chat.Clone: litertlm: conversation_clone failed

This is an upstream limitation: as of LiteRT-LM v0.12.0 the
LiteRT executor used by Gemma 4 on CPU and GPU returns
Unimplemented from Session.Clone. The wrapper binds the C
symbol correctly; Clone will work once upstream lands
Session.Clone in the LiteRT executor.
```

The example exits cleanly (exit code 0) in that path so CI can
distinguish a known upstream gap from a regression.

## Lifecycle

- The parent Chat owns the underlying `ConversationConfig`. Clones
  reuse the parent's config and release only their `Conversation` on
  `Close`.
- The parent Chat must outlive every clone derived from it. Closing
  the parent invalidates the configuration that backs each clone's
  Conversation.
- Tools and the dispatch-hop limit are shared across parent and
  clones. The shared `map[string]ToolDefinition` is read-only after
  Chat construction; concurrent reads are safe.

## Files

- [`main.go`](./main.go) — single-file example, no extra dependencies.
