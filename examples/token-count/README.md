# Example: Tracking KV-Cache Tokens with Chat.TokenCount

Runs a multi-turn conversation and prints how many tokens the
conversation holds in its KV cache after each turn, as a fraction of the
engine's max-token budget.

## What this example shows

- `Chat.TokenCount()` returning the live KV-cache token count (prefill +
  decode, across all turns). The number grows with each turn.
- No `WithBenchmarkEnabled` — `TokenCount` reads the KV-cache size
  directly, so it works on any Client.
- Using the count to project a chat against `WithMaxTokens` for budget /
  compaction decisions.

For a per-turn prefill / decode breakdown instead, enable benchmark
collection and read `Conversation.BenchmarkInfo()`.

## Prerequisites

1. LiteRT-LM shared library files (v0.13.1+) staged in `LITERTLM_LIB`.
2. A `.litertlm` chat-tuned model (e.g. Gemma 4).

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/token-count \
    -model /abs/path/to/gemma-4-E2B-it.litertlm
```

| Flag       | Default        | Notes                                                      |
| ---------- | -------------- | ---------------------------------------------------------- |
| `-model`   | (required)     |                                                            |
| `-max`     | `4096`         | Engine max tokens; the count is reported as a % of this.   |
| `-backend` | `"cpu"`        |                                                            |
| `-lib`     | `$LITERTLM_LIB`|                                                            |

## Expected output

```
    [start] KV cache: 0 / 4096 tokens (0.0% of budget)
user> Name three primary colors.
bot>  Red, yellow, and blue.
    [after turn] KV cache: 33 / 4096 tokens (0.8% of budget)

user> Now name three secondary colors.
bot>  Orange, green, and violet.
    [after turn] KV cache: 55 / 4096 tokens (1.3% of budget)

user> Summarize this conversation in one sentence.
bot>  The conversation involved naming the primary and secondary colors.
    [after turn] KV cache: 82 / 4096 tokens (2.0% of budget)
```

Requires LiteRT-LM v0.13.1 or newer (the underlying
`litert_lm_conversation_get_token_count` symbol).
