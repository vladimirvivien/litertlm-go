# cache-warmup

Contrast a cold and a warm `WithCacheDir` load.

## What this exercises

`WithCacheDir(dir)` points the engine at a directory it may use for
artefact caching — XNNPACK kernel compile cache on CPU, mldrift
program cache on GPU. The first load against an empty dir compiles
the kernels and writes the artefacts; a subsequent load against the
same dir reuses them and skips the compile step.

The example issues a one-shot `Generate` after each load so any
kernel work the engine defers until the first prefill is included in
the timing.

## Prerequisites

- `.litertlm` model file. Gemma 4 family works.
- LiteRT-LM shared libraries on disk. Pass `-lib <dir>` or set
  `LITERTLM_LIB=<dir>`.

## Flags

| Flag | Default | Description |
|---|---|---|
| `-model` | (required) | Path to `.litertlm` model file. |
| `-lib` | `$LITERTLM_LIB` | Directory holding the shared libraries. |
| `-backend` | `cpu` | `cpu` or `gpu`. |
| `-cache-dir` | (fresh temp dir) | Directory passed to `WithCacheDir`. Empty creates a temp dir removed at exit. |
| `-keep-cache` | `false` | Skip cleanup of an auto-created temp dir. |
| `-prompt` | `"The capital of France is"` | One-shot Generate after each load. |

## Run

```sh
go run ./examples/cache-warmup -model "$MODEL" -backend cpu
```

## Expected output

Observed on Gemma 4 E2B, CPU, Windows 11, hot NVMe:

```
cache dir: C:\Users\...\Temp\litertlm-cache-warmup-...

=== Run 1 (cold) ===
  litertlm.New: 3.4894298s
  Generate:     563.1419ms
  TimeToFirstToken: 450.364833ms
  Prefill: 15.2 tok/s (6 tokens)
  Decode:  17.8 tok/s (3 tokens)

=== Run 2 (warm) ===
  litertlm.New: 238.1818ms
  Generate:     569.2116ms
  TimeToFirstToken: 454.105733ms
  Prefill: 15.1 tok/s (6 tokens)
  Decode:  17.6 tok/s (3 tokens)

=== Delta (cold - warm) ===
  litertlm.New: 3.251248s
  Generate:     -6.0697ms

artefacts written to ...:
  gemma-4-E2B-it.litertlm.xnnpack_cache_13422901844_2588147712 (788412736 bytes)
```

The cold→warm delta lands almost entirely on `litertlm.New`. Generate
timings stay flat: the cache affects kernel compile, not inference
work. The artefact list shows the XNNPACK file the engine wrote on
the cold run and reused on the warm run.

## Notes

- The artefact's name encodes the model's weight buffer hashes, so
  the same cache dir can hold artefacts for multiple models.
- On a backend with no compile caching (or one whose kernels are
  small enough that compile time is negligible) the two timings can
  be similar and no artefact is written. The example reports that
  honestly via the `no artefacts written` message.
- The XNNPACK file is large (~750 MB for Gemma 4 E2B). Point
  `-cache-dir` at a known location with free disk before persisting.
