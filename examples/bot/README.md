# Example: `bot` — CLI chat assistant

A tiny but real general-purpose chat assistant. Streams replies
token-by-token, optionally persists conversation memory across runs,
loads a custom system prompt from `SYSTEM.md` when present, and
self-compacts that memory when it would overflow the engine's context
budget.

By default the bot runs **ephemeral** — fast startup, no on-disk
state. Pass `-mem MEM.log` for cross-run memory.

This is also a working tutorial for four corners of the litertlm-go
API in one file:

- `Chat.SendStream` — streaming output with the chat template applied.
- `Client.NewChat` / `Chat.Send` — the same API used here for the
  one-shot summarization step.
- `Client.TokenLength` — exact token counting for budgeting decisions.
- A real "bot manages its own memory" pattern using the bot's own
  inference to compact its history.

## Prerequisites

1. LiteRT-LM shared library files staged in `LITERTLM_LIB`.
2. A `.litertlm` chat-tuned model (e.g. Gemma 4 E2B or E4B).

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
  go run ./examples/bot \
  --model /abs/path/to/gemma-4-E2B-it.litertlm
```

Talk to it (ephemeral, no persistence):

```
warming up... done in 660ms
loaded gemma-4-E2B-it.litertlm  (memory: ephemeral, 70 system tokens, budget: 4096 max - 1024 reply reserve)

〉 what's a good name for a fishing boat?

🤖 Some options: Sea Sprite, Reel Deal, Tide Runner...

〉 exit
```

Exit the REPL with `exit`, `/exit`, `Ctrl+D` (EOF), or `Ctrl+C` (SIGINT). All four shut down cleanly.

For cross-run memory, pass `-mem MEM.log`:

```bash
go run ./examples/bot --model ... --mem MEM.log
```

```
warming up... done in 440ms
loaded gemma-4-E2B-it.litertlm  (memory: MEM.log, 0 turns, 70 system tokens, budget: 4096 max - 1024 reply reserve)

〉 i like Tide Runner. remember that.

🤖 Got it — Tide Runner.

〉 exit
```

Re-run with the same `-mem MEM.log` and the bot remembers:

```
warming up... done in 2.8s
loaded gemma-4-E2B-it.litertlm  (memory: MEM.log, 2 turns, 159 system tokens, budget: 4096 max - 1024 reply reserve)

〉 what was that name?

🤖 Tide Runner.
```

Note that warmup time grows with persisted memory size — the handshake
prefills the system prompt + memory at startup, so you pay the
prefill cost once instead of on every first reply. See **Sizing
`-max`** below for guidance on long-session configurations.

## Startup sequence

`litertlm.New` loads the model weights, but the engine defers some
fixed setup cost (TFLite executor lazy init, allocator preallocation,
model-file mmap page-in) until the first inference. Without
intervention, the user pays this on their first prompt — they type,
hit enter, and wait noticeably longer than for subsequent replies.

The bot frontloads this with a one-shot warmup inference against a
throwaway temp `Chat`:

```
Waking up ⏱️... done in 982ms
```

Then the prompt appears and the first real reply runs at steady-state
speed. With ephemeral memory (the default) the system prompt is small,
so warmup is sub-second and the first user turn pays only its own
prefill + decode.

If you enable persistence with `-mem MEM.log` and your transcript
grows to thousands of tokens, the first user turn after startup pays
the full system + memory prefill cost (because warmup uses a temp
Chat, not the main one). Compaction keeps that cost bounded.

## Files

| File | Read at startup? | Written during run? | Purpose |
|------|:---:|:---:|---------|
| `SYSTEM.md` | yes (if present) | no | Custom system prompt |
| `MEM.log`   | yes (if present) | yes (each turn) | Rolling conversation memory |

Both are plain text — hand-edit, version-control, or delete between
runs.

## System prompt resolution

First match wins:

1. `-system "..."` flag
2. `-system-file path` flag
3. `SYSTEM.md` in cwd
4. Built-in default: *"You are a helpful assistant. Keep answers
   concise unless asked for detail."*

## `MEM.log` format

A line-oriented, append-only transcript. Each turn is a 4-backtick
fenced block:

````
> user 2026-05-10T14:30:12Z````
explain why the sky is blue
````
> bot 2026-05-10T14:30:15Z````
The sky appears blue due to Rayleigh scattering...
````
````

- Header: `> role timestamp````` (`role` is `user` or `bot`,
  timestamp is RFC 3339 UTC).
- Body is everything between the opening fence and the next bare
  closing fence. Preserved verbatim — embedded triple-backtick code
  blocks are fine; only a line of four-or-more backticks alone closes
  the body.
- Token overhead is ~9 tokens per turn (header + closing fence).

## Compaction

Before each turn, the bot computes:

```
projected = TokenLength(systemPrompt + memory) + TokenLength(userInput) + replyReserve
threshold = maxTokens × compactAt
```

If `projected > threshold`, it compacts before sending the user's
message. Compaction:

1. Closes the main chat (the C engine permits one active session at a time).
2. Opens a temporary chat with a dedicated summarization system prompt.
3. Sends the full transcript and captures the summary.
4. Overwrites `MEM.log` with the summary as a single `bot` block.
5. Reopens the main chat with the new effective system prompt.

The user sees:

```
[compacting memory: 47 turns, 3214 tokens → 612 tokens]
```

Compaction uses one prefill cost; subsequent turns return to normal
speed.

## Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `-model` | (required) | Path to `.litertlm` chat-tuned model |
| `-lib` | `$LITERTLM_LIB` | LiteRT-LM library directory |
| `-backend` | `cpu` | `cpu` or `gpu` |
| `-system` | (none) | Inline system prompt (overrides files) |
| `-system-file` | (none) | Path to system-prompt file (overrides `SYSTEM.md`) |
| `-mem` | `""` (ephemeral) | Path to a memory file to persist conversation across runs (e.g. `MEM.log`). Empty = no on-disk memory. |
| `-max` | `4096` | Engine max tokens (prompt + output). Raise toward the model's context ceiling (32K for Gemma 4) for longer chats; KV cache grows linearly with this value. |
| `-prompt` | (none) | One-shot mode; sends this and exits |
| `-speculative` | `false` | Enable multi-token-prediction (E4B-class only on CPU) |
| `-reset` | `false` | Truncate the memory file before starting (requires `-mem`) |
| `-compact-now` | `false` | Force a compaction at startup (requires `-mem`) |
| `-compact-at` | `0.80` | Compact when projected > this fraction of `-max` |
| `-reply-reserve` | `1024` | Tokens reserved for the model's reply when budgeting |

## Sizing `-max`

The LiteRT-LM C API does not expose the model's intrinsic context
ceiling (no `engine_get_max_seq_len`-style call). `-max` is therefore
a static caller choice, not a percentage of the model's ceiling. The
default of `8192` is a comfortable mid-point for Gemma 4 on a typical
laptop:

| `-max` | KV cache (Gemma 4 E2B) | KV cache (Gemma 4 E4B) | Working memory after `-reply-reserve 1024` |
|-------:|---------------:|---------------:|---------------------------:|
| 4096   | ~0.75 GB       | ~1.5 GB        | ~3000 tokens (~25 turns) |
| 8192   | ~1.5 GB        | ~3 GB          | ~6500 tokens (~60 turns) |
| 16384  | ~3 GB          | ~6 GB          | ~14500 tokens (~140 turns) |
| 32768  | ~6 GB          | ~12 GB         | ~30500 tokens (~300 turns) — Gemma 4 ceiling |

Raise toward the ceiling when you have RAM and want long sessions
between compactions; lower when memory is tight.

## Limitations

- **`-max` must be generous enough for the compaction step.** The
  compaction call sends the full transcript through a fresh chat with
  its own system prompt; if `transcript + compactionSystemPrompt +
  replyReserve` exceeds `-max`, compaction will fail with a
  `DYNAMIC_UPDATE_SLICE` tensor error from the prefill. Default
  `-max 4096` handles ~3000-token transcripts comfortably; lower it
  only when you've measured.
- **Memory is single-tenant.** `MEM.log` lives in the cwd; running two
  bots in the same directory will interleave their writes. Use `-mem`
  to point each instance at its own file.
- **No memory rotation.** Compaction overwrites `MEM.log` in place. If
  the summary loses something important, hand-edit before the next
  run.
- **Body terminator caveat.** A model-emitted line of four-or-more
  backticks alone would terminate the stored body early. In practice,
  models emit three-backtick fences for code blocks, so this is rare —
  but worth knowing if you ever see truncated reloads.

## Custom system prompt

Drop a `SYSTEM.md` in the directory you run the bot from:

```
$ cat SYSTEM.md
You are a senior Go reviewer. Keep replies short, point at line numbers,
and refuse to write code unless explicitly asked.

$ go run ./examples/bot --model ...
```

Or pass `-system-file path/to/file.md` to use any file.

## What this example does NOT do

- Authentication, tool use, multimodal input — see `examples/autotool`,
  `examples/vision`, `examples/extract` for those.
- Streaming compaction — the summarization step is non-streaming
  (`Chat.Send`) because the transcript fits in one shot.
- Concurrent sessions — the C engine permits one at a time, and the
  bot is single-user by design.
