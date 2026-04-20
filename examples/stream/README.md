# stream — token-by-token streaming via Go channel

Same model, same lifecycle as [`hello`](../hello/), but consumes the
generation result as a stream of `StreamChunk` values delivered through a
Go channel — so output appears progressively as the model decodes.

## What this example shows

- `Session.GenerateContentStreamCh(inputs) <-chan litertlm.StreamChunk`
  — the channel-idiomatic streaming wrapper.
- How `StreamChunk{Text, Final, Err}` is used: print `Text` as it arrives,
  detect end-of-stream via `Final`, propagate runtime failures via `Err`.
- That under the hood, a single `purego.NewCallback` trampoline dispatches
  C-side callbacks back into Go on a background thread (see
  `pkg/litertlm/stream.go`).

## Prerequisites

Identical to [`hello`](../hello/):

1. Native shared library + `libGemmaModelConstraintProvider.*` staged in
   a directory pointed to by `LITERTLM_LIB`.
2. A `.litertlm` model file.
3. Go 1.22+.

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/stream \
    -model /abs/path/to/gemma-4-E2B-it.litertlm \
    -prompt "Write a short haiku about the sea."
```

| Flag        | Default                                   |
| ----------- | ----------------------------------------- |
| `-model`    | (required)                                |
| `-prompt`   | `"Write a short haiku about the sea."`    |
| `-backend`  | `"cpu"`                                   |
| `-max`      | `1024`                                    |
| `-lib`      | `$LITERTLM_LIB`                           |

## Expected output

The text appears chunk-by-chunk, terminated by a newline once `Final=true`:

```
Blue waves crash on shore,
Salt wind whispers secrets deep,
Ocean calls to soul.
```

## Notes

- The stream is delivered on a background thread inside the C library;
  `GenerateContentStreamCh` hides this with a goroutine + buffered channel,
  so on the consumer side it looks like an ordinary `for range`.
- The function blocks the launching goroutine until the final chunk has
  been sent. To run it concurrently with other work, just launch the
  range loop inside its own goroutine.
- If you want the raw callback form (no channel, no goroutine), call
  `Session.GenerateContentStream(inputs, func(StreamChunk){…})` directly.
- Like `hello`, this uses the low-level Session API and sends the prompt
  *without* chat-template wrapping. Most "completion-style" prompts work;
  for chat-style prompts use the [`chat`](../chat/) example.
