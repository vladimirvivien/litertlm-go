# Example: Multimodal structured extraction

Demonstrates `litertlm.GenerateDataMulti[T]` — type-safe extraction
from a multimodal input (image + text instruction). Follows the
extraction with a self-comparison pass against a Markdown
ground-truth sidecar.

## What this example shows

- Loading an image with `litertlm.ImageFromFile`.
- `GenerateDataMulti[Scene]` returns a populated `*Scene` directly
  from the model's image understanding — the JSON-shape instruction
  is injected automatically.
- `WithRetries(n)` re-runs on parse failure (model emitted
  almost-JSON or wrapped its answer in markdown fences).
- `*GenerateDataError` distinguishes parse failures (model returned
  text that couldn't unmarshal) from generate failures (FFI / ctx
  cancellation).
- A second-pass self-evaluation: the model compares its extracted
  JSON to the reference description and reports differences.

## Prerequisites

1. LiteRT-LM shared library files staged in `LITERTLM_LIB`.
2. A multimodal `.litertlm` model (e.g. a Gemma 4 multimodal variant).
3. `WithVisionBackend(...)` set — the example defaults to `"cpu"`.

## Run

From inside this directory:

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run . \
    -model /abs/path/to/gemma-4-mm.litertlm
```

From the repo root, point `--testdata` at the shared directory:

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/extract \
    -model /abs/path/to/gemma-4-mm.litertlm \
    -testdata ./examples/testdata
```

## Flags

| Flag              | Default                                  | Notes                                       |
| ----------------- | ---------------------------------------- | ------------------------------------------- |
| `-model`          | (required)                               | Path to a multimodal `.litertlm` model.     |
| `-lib`            | `$LITERTLM_LIB`                          |                                             |
| `-backend`        | `"cpu"`                                  | Text-side backend.                          |
| `-vision-backend` | `"cpu"`                                  | Vision-side backend.                        |
| `-testdata`       | `"../testdata"`                          | Directory holding `<name>.<ext>` + `<name>.md`. |
| `-name`           | `"img2"`                                 | Basename of the image and `.md` sidecar.    |
| `-prompt`         | `"Extract a one-sentence scene description and a list of distinct objects visible in the image."` | Instruction sent with the image. |
| `-retries`        | `2`                                      | Max retry attempts on parse failure.        |
| `-max-tokens`     | `4096`                                   | Engine token budget. Vision needs ≥4096 — image patches expand into many tokens. |

## Expected output

```
=== Image: ../testdata/img2.png ===

=== Extracted Scene ===
{
  "description": "<one-line scene summary>",
  "objects": ["object 1", "object 2", "..."]
}

=== Reference (ground truth) ===
<contents of img2.md>

=== Model self-comparison ===
<the model's discussion of agreements and differences>
```

## Notes

- **The Scene struct is generic on purpose.** Vision-language models
  are typically more reliable producing a free-form description plus an
  object list than a domain-specific schema with required fields.
  Define your own `T` to extract any shape — e.g.,
  `Recipe { Title, Ingredients, Steps }` for a recipe-card photo,
  `BusinessCard { Name, Phone, Email, Company }` for a contact image.
- **Parse failures.** If the model wraps its JSON in `` ```json ... ``` ``
  fences or appends prose, the tolerant extractor strips them. If it
  invents fields outside the schema, `*GenerateDataError` with
  `Phase == "parse"` carries the raw output for inspection.
- **Custom images.** Drop a `myimg.jpg` and `myimg.md` into
  `examples/testdata/` and pass `-name myimg`.
