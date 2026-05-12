# Example: Multimodal audio Q&A

Transcribes (or answers a question about) a short audio clip, then
optionally asks the model to assess how closely its own transcript
matches a reference loaded from disk.

## What this example shows

- Loading audio with `litertlm.AudioFromFile` (MIME inferred from
  extension) or `litertlm.AudioWithMime` (explicit MIME for files
  whose extension does not match the container).
- Selecting the audio encoder backend with
  `litertlm.WithAudioBackend(...)` — independent of the text-side
  `WithBackend(...)`.
- Calling `client.GenerateMulti(ctx, []Part{Audio, Text})` against a
  multimodal `.litertlm` model.
- An optional follow-up text-only chat turn that scores its own
  transcript against the reference.

## Prerequisites

1. LiteRT-LM shared library files staged in `LITERTLM_LIB`.
2. A multimodal `.litertlm` model that includes an audio encoder
   (e.g. a Gemma 4 multimodal variant).
3. A short audio clip in a supported format (`.wav`, `.mp3`, `.ogg`,
   `.flac`, `.m4a`, `.aac`).

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/audio \
    -model /abs/path/to/gemma-4-mm.litertlm \
    -audio /abs/path/to/clip.wav
```

With a reference transcript for the optional alignment step:

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/audio \
    -model      /abs/path/to/gemma-4-mm.litertlm \
    -audio      /abs/path/to/clip.wav \
    -transcript /abs/path/to/clip.txt
```

## Flags

| Flag              | Default                                              | Notes                                                                |
| ----------------- | ---------------------------------------------------- | -------------------------------------------------------------------- |
| `-model`          | (required)                                           | Path to a multimodal `.litertlm` model.                              |
| `-audio`          | (required)                                           | Audio file (`.wav` / `.mp3` / `.ogg` / `.flac` / `.m4a` / `.aac`).   |
| `-audio-mime`     | derived from extension                               | Override when the file extension does not match the actual MIME.     |
| `-transcript`     | unset                                                | Optional reference transcript; enables the alignment-assessment step. |
| `-lib`            | `$LITERTLM_LIB`                                      |                                                                      |
| `-backend`        | `"cpu"`                                              | Text-side backend.                                                   |
| `-audio-backend`  | `"cpu"`                                              | Audio encoder backend.                                               |
| `-prompt`         | `"Transcribe the speech in this audio clip verbatim."` | Swap in `"Summarize…"`, `"Identify the speaker's tone…"`, etc.     |
| `-max-tokens`     | `4096`                                               | Engine token budget; audio frames expand into many tokens like image patches. |

## Expected output

Without `-transcript`:

```
=== Audio: /abs/path/to/clip.wav (audio/wav) ===

=== Model output ===
<verbatim transcript, or whatever the prompt asked for>
```

With `-transcript`:

```
=== Audio: /abs/path/to/clip.wav (audio/wav) ===

=== Model output ===
<the model's own transcript>

=== Reference transcript ===
<contents of -transcript file>

=== Alignment assessment ===
<2-3 sentences from the model on how closely its transcript matches the reference>
```

## Notes

- **Audio and text backends are independent.** LiteRT-LM keeps the
  text path and the audio encoder in separate slots, so you can run
  text on GPU and the audio encoder on CPU (or vice-versa) by setting
  the two flags differently.
- **Token budget.** Audio waveforms expand into many tokens on the
  way into the model — like image patches do for vision. `4096` is
  a safe floor for short clips; longer clips may need more.
- **Alignment is a qualitative signal.** The optional assessment
  step is a normal text generation; the model can be wrong in either
  direction. Treat it as a sanity check, not an evaluation harness.
- **Why two load paths?** `AudioFromFile` is the common case — MIME
  is inferred from extension. `AudioWithMime` is for raw bytes or
  files whose extension does not match the container.
