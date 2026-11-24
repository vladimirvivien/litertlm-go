# Session Checkpoints, Cloning & Async Decode (`examples/checkpoints`)

Demonstrates low-level KV-cache state management using `Session.SaveCheckpoint`,
`Session.RewindToCheckpoint`, `Session.Clone`, and `Session.RunDecodeAsyncCh`.

Instead of re-prefilling the shared prefix for each candidate continuation, the
example prefills the prompt prefix once, snapshots a checkpoint, clones an
independent `Session`, rewinds to branch into a second completion, and streams
every decode via `RunDecodeAsyncCh`.

## Run

```bash
go run ./examples/checkpoints \
    -get-lib v0.19.1 \
    -get-model litert-community/gemma3-1b-it-int4
```
