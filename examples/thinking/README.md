# Thinking Configuration (`examples/thinking`)

Demonstrates reasoning controls using `WithThinking(ThinkingConfig{Enabled, TokenBudget})`,
`WithFilterChannelContentFromKVCache(true)`, and per-turn `WithTurnThinking`.

## Run

```bash
go run ./examples/thinking \
    -get-lib v0.19.1 \
    -model /abs/path/to/Qwen3-0.6B.litertlm
```
