# Example: Multimodal vision Q&A

Asks the model to describe an image, then asks it to assess how well
its description aligns with a reference description loaded from disk.

## What this example shows

- Loading an image with `litertlm.ImageFromFile` (MIME inferred from
  extension).
- Calling `client.GenerateMulti(ctx, []Part{Image, Text})` against a
  multimodal `.litertlm` model.
- A follow-up text-only `client.Generate` call that produces a brief
  alignment assessment between the model's description and the
  reference.

## Prerequisites

1. LiteRT-LM shared library files staged in `LITERTLM_LIB`.
2. A multimodal `.litertlm` model (e.g. a Gemma 4 multimodal variant).
3. `WithVisionBackend(...)` set — the example defaults to `"cpu"`.

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/vision \
    -model       /abs/path/to/gemma-4-mm.litertlm \
    -image       /abs/path/to/photo.png \
    -description /abs/path/to/photo.md
```

## Flags

| Flag              | Default                                  | Notes                                       |
| ----------------- | ---------------------------------------- | ------------------------------------------- |
| `-model`          | (required)                               | Path to a multimodal `.litertlm` model.     |
| `-image`          | (required)                               | Path to the image file (jpg/png/webp/gif/bmp). |
| `-description`    | (required)                               | Path to the reference description file.     |
| `-lib`            | `$LITERTLM_LIB`                          |                                             |
| `-backend`        | `"cpu"`                                  | Text-side backend.                          |
| `-vision-backend` | `"cpu"`                                  | Vision-side backend.                        |
| `-prompt`         | `"Describe this image in 2-3 sentences."`| Instruction sent with the image.            |
| `-max-tokens`     | `4096`                                   | Engine token budget; vision needs ≥4096 because image patches expand into many tokens. |

## Expected output

```
=== Image: /abs/path/to/photo.png ===

=== Model description ===
<the model's free-text description of the scene>

=== Reference description ===
<contents of the description file>

=== Alignment assessment ===
<2-3 sentences from the model on how well its description matches the reference>
```

## Notes

- **Alignment is a qualitative signal.** The assessment step is a
  normal text generation; the model can be wrong in either direction.
  Treat it as a sanity check, not an evaluation harness.
- **Vision and text backends are independent.** LiteRT-LM keeps text
  and vision backends in separate slots; you can run text on GPU and
  the vision tower on CPU (or vice-versa) by setting the two flags
  differently.
