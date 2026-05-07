# Example: Multimodal vision Q&A

Demonstrates `Client.GenerateMulti` — a single high-level call that
mixes image and text Parts — followed by a text-only `Client.Generate`
self-evaluation against a Markdown ground-truth sidecar.

## What this example shows

- Loading an image with `litertlm.ImageFromFile` (MIME inferred from
  extension).
- Calling `client.GenerateMulti(ctx, []Part{Image, Text})` against a
  multimodal `.litertlm` model.
- A second-pass self-evaluation: the model re-reads its own output
  alongside an external reference and produces a structured
  comparison.

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
    go run ./examples/vision \
    -model /abs/path/to/gemma-4-mm.litertlm \
    -testdata ./examples/testdata
```

## Flags

| Flag              | Default                                  | Notes                                      |
| ----------------- | ---------------------------------------- | ------------------------------------------ |
| `-model`          | (required)                               | Path to a multimodal `.litertlm` model.    |
| `-lib`            | `$LITERTLM_LIB`                          |                                            |
| `-backend`        | `"cpu"`                                  | Text-side backend.                         |
| `-vision-backend` | `"cpu"`                                  | Vision-side backend.                       |
| `-testdata`       | `"../testdata"`                          | Directory holding `<name>.<ext>` + `<name>.md`. |
| `-name`           | `"img1"`                                 | Basename of the image and `.md` sidecar.   |
| `-prompt`         | `"Describe this image in 2-3 sentences."`| Instruction sent with the image.           |
| `-max-tokens`     | `4096`                                   | Engine token budget. Vision needs ≥4096 — image patches expand into many tokens. |

The example tries `<name>.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.bmp`
in order until it finds a file. The matching `<name>.md` is loaded
verbatim and used as the reference description in step 2.

## Expected output

```
=== Image: ../testdata/img1.png ===

=== Model description ===
<the model's free-text description of the scene>

=== Reference (ground truth) ===
<contents of img1.md>

=== Model self-comparison ===
<the model's discussion of agreements and differences>
```

## Notes

- **Self-comparison reliability.** The compare step is a normal text
  generation — the model can be wrong in either direction (claim
  agreement when it disagrees, or vice versa). Treat it as a
  qualitative signal, not an evaluation harness.
- **Custom images.** Drop a `myscene.jpg` and `myscene.md` into
  `examples/testdata/` and pass `-name myscene`.
- **Vision backend.** The C side of LiteRT-LM keeps text and vision
  backends in independent slots; you can run text on GPU and the
  vision tower on CPU (or vice-versa) by setting the two flags
  differently.
