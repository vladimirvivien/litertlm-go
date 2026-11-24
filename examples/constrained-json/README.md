# Grammar-Constrained Output (`examples/constrained-json`)

Demonstrates token-level constrained decoding using `WithConstrainedDecoding(true)`,
`WithConstraintProvider(litertlm.ConstraintProviderLLGuidance)`, and per-turn
constraints (`ConstraintJSONSchema` and `ConstraintRegex`).

## Run

```bash
go run ./examples/constrained-json \
    -get-lib v0.19.1 \
    -get-model litert-community/gemma3-1b-it-int4
```
