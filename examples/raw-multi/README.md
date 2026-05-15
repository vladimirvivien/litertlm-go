# raw-multi

Run the same image + caption-text input through each of the three
`*Multi` siblings on `Client` to surface the call-shape and
return-type differences:

| Call | Returns |
|---|---|
| `Client.GenerateMulti(ctx, parts, opts...)` | `string` |
| `Client.GenerateMultiStream(ctx, parts, opts...)` | `iter.Seq2[Chunk, error]` |
| `Client.GenerateMultiResponse(ctx, parts, opts...)` | `*Response` |

## Prerequisites

- `.litertlm` model with vision support (e.g. Gemma 4).
- LiteRT-LM shared libraries on disk. Pass `-lib <dir>` or set
  `LITERTLM_LIB=<dir>`.
- An image file (`png`, `jpg`, `webp`).

## Flags

| Flag | Default | Description |
|---|---|---|
| `-model` | (required) | Path to `.litertlm` model file. |
| `-lib` | `$LITERTLM_LIB` | Directory holding the shared libraries. |
| `-backend` | `cpu` | LLM inference backend. |
| `-vision-backend` | `cpu` | Vision encoder backend. |
| `-image` | (required) | Path to the image file. |
| `-prompt` | `"Describe this image in one sentence."` | Caption prompt. |
| `-max-tokens` | `128` | Per-call decode cap. |

## Run

```sh
go run ./examples/raw-multi \
    -model "$MODEL" \
    -image examples/testdata/img1.png
```

## Observed output

Gemma 4 E2B, CPU, `examples/testdata/img1.png`:

```
image:  examples/testdata/img1.png
prompt: Describe this image in one sentence.

=== GenerateMulti (synchronous) ===
A warm, rustic wooden table is set with a lamp, a mug of coffee, and a pair of glasses, set against a muted green wall with a window showing some greenery.

=== GenerateMultiStream ===
A warm, rustic wooden table is set with a lamp, a mug of coffee, and a pair of glasses, set against a muted green wall with a window showing some greenery.
(chunks delivered: 37)

=== GenerateMultiResponse ===
A warm, rustic wooden table is set with a lamp, a mug of coffee, and a pair of glasses, set against a muted green wall with a window showing some greenery.
token length: 36
time to first token: 3.591297021s
prefill: 78.9 tok/s
decode:  23.8 tok/s
```

Vision encoding is paid three times (once per call). Use a small
image for quick iterations.

## See also

- `NewBinaryInput(t InputDataType, b []byte) InputData` is the
  Session-level equivalent for raw image/audio bytes. Use it with
  `Session.GenerateContent` on the low-level path
  (`examples/prefill-decode/`, `examples/conversation-lowlevel/`).
- `examples/vision/` and `examples/extract/` exercise
  `GenerateData[T]` over the same image-plus-text input shape.
