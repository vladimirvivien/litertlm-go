# bot

A CLI chat assistant. Each run is a REPL session that streams replies
token-by-token; conversation memory persists across runs in `MEM.log`,
and the bot uses its own inference to compact that memory when it
would overflow the engine's context budget. The REPL accepts image
and audio attachments via slash commands; the persisted transcript
records them as text annotations.

By default the bot runs **ephemeral** — fast startup, no on-disk
state. Pass `-mem MEM.log` for cross-run memory.

## Prerequisites

- `.litertlm` chat-tuned model (e.g. Gemma 4 E2B or E4B).
- LiteRT-LM shared libraries on disk. Pass `-lib <dir>` or set
  `LITERTLM_LIB=<dir>`.

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
  go run ./examples/bot \
  -model /abs/path/to/gemma-4-E2B-it.litertlm
```

Ephemeral session:

```
Waking up ⏱️... done in 1.03s
🤖 gemma-4-E2B-it.litertlm
💾 ephemeral · 📝 13 system tokens
📐 4096 max · 1024 reply reserve

〉 what's a good name for a fishing boat?

🤖 Some options: Sea Sprite, Reel Deal, Tide Runner...

〉 exit
```

Exit the REPL with `exit`, `/exit`, `Ctrl+D` (EOF), or `Ctrl+C` (SIGINT).

For cross-run memory, pass `-mem MEM.log`:

```bash
go run ./examples/bot -model ... -mem MEM.log
```

Re-run with the same `-mem MEM.log` and the bot remembers via the
`WithInitialMessages` bridge — prior turns are fed to the chat
template with proper `user` / `assistant` role markers.

## Slash commands

| Command | Effect |
|---|---|
| `/attach <path>` | Queue a media file (image or audio) for the next message. Extension drives the loader. |
| `/attachments` | List queued attachments. |
| `/clear` | Drop queued attachments. |
| `/help` | List commands. |
| `exit` / `/exit` | Quit. |

Attachments are consumed after each multimodal send. Re-attach
before the next multimodal turn.

```
〉 /attach photo.jpg
attached: photo.jpg (image)

〉 /attach voice.wav
attached: voice.wav (audio)

〉 What's in the picture and what does the voice say?

🤖 ...
```

## Multimodal turn semantics

Turns with queued attachments go through `Client.GenerateMultiStream`
instead of the persistent `Chat.SendStream` path. The C engine
permits one active session at a time, so the Chat is closed before
the multimodal call and reopened afterwards — the next text turn
then sees the new memory entry via `WithInitialMessages`. The reopen
re-prefills the system prompt + prior turns once per multimodal
turn.

## File-type detection

| Extension | Loader |
|---|---|
| `.png .jpg .jpeg .webp` | `ImageFromFile` |
| `.wav .mp3 .flac .ogg .opus .m4a .aac` | `AudioFromFile` |
| `.mp4 .mov .webm .avi .mkv` | rejected with a hint: extract a frame (`ffmpeg -i input.mp4 -ss 5 frame.png`) or the audio track (`ffmpeg -i input.mp4 audio.wav`) and `/attach` that |
| anything else | rejected with the supported-extension list |

## Memory bridge (`WithInitialMessages`)

`MEM.log` is fed to the chat as a base system prompt plus a
`WithInitialMessages` slice. Each saved turn becomes one
`litertlm.Message` with role `user` or `assistant` (the persisted
role `bot` maps to `assistant`).

A lone leading `bot` turn is treated as a **compaction summary** —
it is injected into the system prompt rather than fed as an
assistant message, so the message stream stays a clean
user/assistant alternation that the chat template expects.

Multimodal turns are persisted with text annotations on the user
side:

````
> user 2026-05-15T22:14:26Z````
[image: /abs/path/photo.jpg]
[audio: /abs/path/voice.wav]
What's in the picture and what does the voice say?
````
> bot 2026-05-15T22:14:26Z````
The picture shows ... The voice says, "..."
````
````

On the next run the model sees the `[image: ...]` / `[audio: ...]`
markers as text. The bytes are not stored; conversation continuity
is preserved at the text level.

## Files

| File | Read at startup? | Written during run? | Purpose |
|------|:---:|:---:|---------|
| `SYSTEM.md` | yes (if present) | no | Custom system prompt. |
| `MEM.log` | yes (if present) | yes (each turn) | Rolling conversation memory. |

Both are plain text — hand-edit, version-control, or delete between
runs.

## System prompt resolution

First match wins:

1. `-system "..."` flag
2. `-system-file path` flag
3. `SYSTEM.md` in the working directory
4. Built-in default.

When a compaction summary is present, it is appended to the
resolved base prompt at chat-open time.

## `MEM.log` format

Line-oriented, append-only. Each turn is a 4-backtick fenced block:

````
> user 2026-05-10T14:30:12Z````
explain why the sky is blue
````
> bot 2026-05-10T14:30:15Z````
The sky appears blue due to Rayleigh scattering...
````
````

- Header: `> role timestamp\`\`\`\`` — `role` is `user` or `bot`,
  timestamp is RFC 3339 UTC.
- Body is everything between the opening fence and the next bare
  closing fence. Three-backtick code fences inside bodies are
  preserved; only a line of four-or-more backticks alone closes the
  body.
- Token overhead is ~9 tokens per turn (header + closing fence).

## Compaction

Before each turn, the bot computes:

```
projected = TokenLength(systemPrompt + summary)
          + TokenLength(history-text)
          + TokenLength(userInput)
          + replyReserve
threshold = maxTokens × compactAt
```

If `projected > threshold`, it compacts before sending the user's
message:

1. Close the main chat (the C engine permits one active session at
   a time).
2. Open a temporary chat with the summarization system prompt.
3. Send the full transcript and capture the summary.
4. Overwrite `MEM.log` with the summary as a single `bot` block.
5. Reopen the main chat — the new `MEM.log` is loaded as a system-
   prompt summary; turns recorded after this point flow through
   `WithInitialMessages`.

```
[compacting memory: 47 turns, 3214 tokens → 612 tokens]
```

## Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `-model` | (required) | Path to `.litertlm` chat-tuned model. |
| `-lib` | `$LITERTLM_LIB` | LiteRT-LM library directory. |
| `-backend` | `cpu` | `cpu` or `gpu`. |
| `-vision-backend` | `cpu` | Vision encoder backend; used when an image is attached. |
| `-audio-backend` | `cpu` | Audio encoder backend; used when an audio file is attached. |
| `-cache-dir` | (engine default: alongside the model file) | `WithCacheDir` value. Writable location for XNNPACK / mldrift compile artefacts. Saves ~3 s on cold-vs-warm restart on Gemma 4 E2B CPU (see `examples/cache-warmup/`). |
| `-attach <path>` | (repeatable) | Pre-attach a media file at startup. Stacks for `-prompt` one-shot or for the first REPL turn. |
| `-system` | (none) | Inline system prompt (overrides files). |
| `-system-file` | (none) | Path to system-prompt file (overrides `SYSTEM.md`). |
| `-mem` | `""` (ephemeral) | Path to a memory file to persist conversation across runs. |
| `-max` | `4096` | Engine max tokens (prompt + output). Raise toward the model's context ceiling (32K for Gemma 4) for longer chats; KV cache grows linearly. |
| `-prompt` | (none) | One-shot mode; sends this and exits. |
| `-speculative` | `false` | Enable multi-token-prediction (E4B-class only on CPU). |
| `-reset` | `false` | Truncate the memory file before starting (requires `-mem`). |
| `-compact-now` | `false` | Force a compaction at startup (requires `-mem`). |
| `-compact-at` | `0.80` | Compact when projected > this fraction of `-max`. |
| `-reply-reserve` | `1024` | Tokens reserved for the model's reply when budgeting. |
| `-temperature` | `0.7` | Sampling temperature; 0 = greedy. |
| `-top-p` | `0.95` | Nucleus sampling cutoff. |
| `-top-k` | `40` | Top-k sampling cutoff. |
| `-seed` | `0` | Sampler RNG seed (0 = nondeterministic). |

## Sizing `-max`

The LiteRT-LM C API does not expose the model's intrinsic context
ceiling. `-max` is therefore a static caller choice, not a fraction
of the model's ceiling. Default `4096` is a comfortable mid-point
for Gemma 4 on a typical laptop.

| `-max` | KV cache (E2B) | KV cache (E4B) | Working memory after `-reply-reserve 1024` |
|-------:|---------------:|---------------:|---------------------------:|
| 4096   | ~0.75 GB       | ~1.5 GB        | ~3000 tokens (~25 turns) |
| 8192   | ~1.5 GB        | ~3 GB          | ~6500 tokens (~60 turns) |
| 16384  | ~3 GB          | ~6 GB          | ~14500 tokens (~140 turns) |
| 32768  | ~6 GB          | ~12 GB         | ~30500 tokens (~300 turns) — Gemma 4 ceiling |

Raise toward the ceiling when RAM is available; lower when it is
tight.

## One-shot mode

```bash
go run ./examples/bot \
  -model "$MODEL" \
  -prompt "Describe this image in one sentence." \
  -attach photo.jpg
```

`-prompt` sends one message and exits. `-attach` stacks on
`-prompt` for multimodal one-shots.

## Limitations

- **`-max` must accommodate the compaction step.** Compaction sends
  the full transcript through a fresh chat with its own system
  prompt; if `transcript + compactionSystemPrompt + replyReserve`
  exceeds `-max`, prefill fails with a `DYNAMIC_UPDATE_SLICE` tensor
  error. `-max 4096` fits ~3000-token transcripts.
- **Memory is single-tenant.** `MEM.log` lives in the working
  directory by default; two bots in the same directory interleave
  writes. Point each instance at its own file with `-mem`.
- **No memory rotation.** Compaction overwrites `MEM.log` in place.
- **Body terminator.** A model-emitted line of four-or-more
  backticks alone terminates the stored body early. Models normally
  emit three-backtick fences for code blocks, so this is rare.
- **Multimodal turns re-prefill the chat.** Each multimodal send
  closes and reopens the Chat session; the system prompt and prior
  turns are re-prefilled once per multimodal turn. Text-only turns
  keep the Chat session warm.
- **Video is not supported.** `/attach` and `-attach` reject video
  extensions with an `ffmpeg` extraction hint.

## What this example does NOT cover

- Tool use — see `examples/autotool/` and `examples/tool-policy/`.
- `GenerateData[T]` typed output — see `examples/structured/` and
  `examples/extract/`.
- Streaming compaction — the summarization step uses non-streaming
  `Chat.Send` because the transcript fits in one call.
- Concurrent sessions — the C engine permits one at a time.
