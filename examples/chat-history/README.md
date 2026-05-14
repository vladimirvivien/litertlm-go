# chat-history

Resume a prior conversation by seeding `Chat` with a transcript.

## What this exercises

| Option | Effect |
|---|---|
| `WithInitialMessages([]Message)` | Pre-seeds the conversation with prior `{role, content}` turns. Appended after the system prompt, before the first `Send`. |
| `WithExtraContext(json)` | Attaches a JSON-object preamble (RAG-style notes) to the conversation. The C side requires a JSON **object** — arrays, scalars, and free text are silently dropped. |
| `WithFilterChannelContentFromKVCache(bool)` | Excludes `<|channel> ... <channel|>` reasoning tokens from the KV cache so they do not persist across turns. |
| `WithMaxToolHops(n)` | Caps the dispatch loop's iterations. Takes effect only when `ManagedTools` are registered (`RegisterTool`); without tools the value is inert. |

## Prerequisites

- `.litertlm` model file. Gemma 4 family works.
- LiteRT-LM shared libraries on disk. Either pass `-lib <dir>` or set
  `LITERTLM_LIB=<dir>`.

## Flags

| Flag | Default | Description |
|---|---|---|
| `-model` | (required) | Path to `.litertlm` model file. |
| `-lib` | `$LITERTLM_LIB` | Directory holding the shared libraries. |
| `-backend` | `cpu` | `cpu` or `gpu`. |
| `-system` | concise-assistant prompt | System message. |
| `-history` | built-in 4-turn transcript | Path to a JSON file with the prior transcript. |
| `-context` | "" | Path to a JSON file with the extra-context object. Empty skips `WithExtraContext`. |
| `-message` | "What did I say my name was…" | New user message sent after history seeding. |
| `-filter-channels` | `false` | When `true`, reasoning-channel tokens are not persisted in the KV cache. |
| `-max-tool-hops` | `0` | Tool-dispatch cap. `0` keeps the library default. |
| `-max-tokens` | `4096` | Engine total token budget. |

## Transcript format

JSON array of `{role, content}` objects. Roles are `user` and `assistant`.

```json
[
  {"role": "user", "content": "Hi, my name is Vlad."},
  {"role": "assistant", "content": "Hello Vlad."},
  {"role": "user", "content": "I prefer concise answers."},
  {"role": "assistant", "content": "Understood."}
]
```

## Extra-context format

The `-context` file must decode to a JSON object. Non-object input is
rejected before reaching the C side. Example:

```json
{
  "notes": [
    "User goes by Vlad.",
    "Project is the Go binding for LiteRT-LM."
  ],
  "today": "2026-05-14"
}
```

## Run

```sh
go run ./examples/chat-history \
  -model "$MODEL" \
  -backend cpu
```

With a custom transcript and RAG preamble:

```sh
go run ./examples/chat-history \
  -model "$MODEL" \
  -history ./mychat.json \
  -context ./notes.json \
  -message "Summarize what we discussed."
```

## Expected output

```
=== Seeded history ===
user>      Hi, my name is Vlad and I'm building a Go binding for LiteRT-LM.
assistant> Hello Vlad. Are you using cgo or a pure-Go FFI approach?
user>      Pure-Go via purego and jupiterrider/ffi. No cgo.
assistant> Nice — that avoids the cross-compilation headaches cgo would introduce.

user>      What did I say my name was, and which FFI libraries am I using?
assistant> Your name is Vlad, and you are using the purego and jupiterrider/ffi libraries.
```

## Notes

- The system prompt is bare content; `Chat` JSON-encodes it into the
  conversation config itself.
- `WithFilterChannelContentFromKVCache(false)` matches the engine
  default; pass `true` only for models whose chat template emits
  `<|channel> ... <channel|>` reasoning blocks.
- `WithMaxToolHops` has no effect without `RegisterTool` entries; see
  `examples/autotool/` for the tool-dispatch loop it gates.
- Initial messages are JSON-marshaled in the wrapper and handed to the
  C-side `messagesJSON` parameter of `NewConversationConfig`. The
  equivalent low-level path is in `examples/conversation-lowlevel/`.
