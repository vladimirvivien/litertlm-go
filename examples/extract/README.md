# Example: Multimodal structured extraction

Pulls a typed `Scene` struct out of an image with
`litertlm.GenerateDataMulti[T]`, prints the JSON, then asks the model
to assess how well the extracted JSON aligns with a reference
description loaded from disk.

## What this example shows

- Loading an image with `litertlm.ImageFromFile`.
- `GenerateDataMulti[Scene]` returns a populated `*Scene` directly
  from the model's image understanding; the JSON-shape instruction is
  injected automatically.
- `WithRetries(n)` re-runs on parse failure (model emits
  almost-JSON or wraps its answer in markdown fences).
- `*GenerateDataError` distinguishes parse failures (model returned
  text that couldn't unmarshal) from generate failures (FFI / ctx
  cancellation).
- A follow-up text-only `client.Generate` call that produces a brief
  alignment assessment between the extracted JSON and the reference.

## Prerequisites

1. LiteRT-LM shared library files staged in `LITERTLM_LIB`.
2. A multimodal `.litertlm` model (e.g. a Gemma 4 multimodal variant).
3. `WithVisionBackend(...)` set — the example defaults to `"cpu"`.

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/extract \
    -model       /abs/path/to/gemma-4-mm.litertlm \
    -image       /abs/path/to/scene.png \
    -description /abs/path/to/scene.md
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
| `-prompt`         | `"Extract a one-sentence scene description and a list of distinct objects visible in the image."` | Instruction sent with the image. |
| `-retries`        | `2`                                      | Max retry attempts on parse failure.        |
| `-max-tokens`     | `4096`                                   | Engine token budget; vision needs ≥4096 because image patches expand into many tokens. |

## Expected output

```
=== Image: /abs/path/to/scene.png ===

=== Extracted Scene ===
{
  "description": "<one-line scene summary>",
  "objects": ["object 1", "object 2", "..."]
}

=== Reference description ===
<contents of the description file>

=== Alignment assessment ===
<2-3 sentences from the model on which elements match, what was missed, and what was added>
```

## Notes

- **The Scene struct is generic on purpose.** Vision-language models
  are typically more reliable producing a free-form description plus
  an object list than a domain-specific schema with required fields.
  Define your own `T` to extract any shape — e.g.,
  `Recipe { Title, Ingredients, Steps }` for a recipe-card photo,
  `BusinessCard { Name, Phone, Email, Company }` for a contact image.
- **Parse failures.** If the model wraps its JSON in `` ```json ... ``` ``
  fences or appends prose, the tolerant extractor strips them. If it
  invents fields outside the schema, `*GenerateDataError` with
  `Phase == "parse"` carries the raw output for inspection.
- **Alignment is a qualitative signal.** The assessment step is a
  normal text generation; treat it as a sanity check, not an
  evaluation harness.
