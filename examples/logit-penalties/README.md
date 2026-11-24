# Per-Turn Logit Penalties (`examples/logit-penalties`)

Demonstrates repetition penalties (`WithRepetitionPenalty`), n-gram repetition
banning (`WithNoRepeatNgram`), and token ID suppression (`WithSuppressTokens`).

## Run

```bash
go run ./examples/logit-penalties \
    -get-lib v0.19.1 \
    -get-model litert-community/gemma3-1b-it-int4
```
