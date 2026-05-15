# logging

Demonstrate the LiteRT-LM log severity floor: set it before `New`
via `SetMinLogLevel`, then change it mid-program to show the floor
is a process-global setting that can be toggled at any time.

## What this exercises

| Surface | Use |
|---|---|
| `LogLevel` | Typed enum for the severity floor. |
| `SetMinLogLevel(LogLevel)` | Set the process-global floor. Sole API for log control. |

Level constants (defined in `pkg/litertlm/enums.go`):

| Constant | Value |
|---|---|
| `LogVerbose` | 0 |
| `LogDebug` | 1 |
| `LogInfo` | 2 |
| `LogWarning` | 3 |
| `LogError` | 4 |
| `LogFatal` | 5 |
| `LogQuiet` | 1000 |

`litertlm.New` does not touch the log level. Without an explicit
`SetMinLogLevel` call, the C side's default of `LogInfo` (verbose)
applies. The other examples in this repo call
`SetMinLogLevel(LogQuiet)` at the top of `main` so their output stays
clean.

## Prerequisites

- `.litertlm` model file.
- LiteRT-LM shared libraries on disk. Pass `-lib <dir>` or set
  `LITERTLM_LIB=<dir>`.

## Flags

| Flag | Default | Description |
|---|---|---|
| `-model` | (required) | Path to `.litertlm` model file. |
| `-lib` | `$LITERTLM_LIB` | Directory holding the shared libraries. |
| `-backend` | `cpu` | `cpu` or `gpu`. |
| `-loglevel` | `info` | Starting level. Accepts names (`verbose`..`quiet`) or numeric values. |
| `-prompt` | `"The capital of France is"` | Issued for each of three Generate calls. |

## Run

```sh
go run ./examples/logging -model "$MODEL" -backend cpu
go run ./examples/logging -model "$MODEL" -backend cpu -loglevel quiet
```

## Observations

Most LiteRT-LM log output comes from `New` (loader, executor init,
magic-number rewriting, accelerator registration). Per-Generate
chatter is minimal at any level. The dominant lever is the floor at
construction time.

On Gemma 4 E2B, CPU, Windows 11, total log lines emitted with three
sequential Generate calls (short prompt):

| Starting level | Total lines |
|---|---|
| `info` (2) | ~360 |
| `quiet` (1000) | ~5 |

Within a `-loglevel info` invocation, the three Generate blocks
decay as `SetMinLogLevel` lowers the floor:

```
=== Generate #1: log level = Info ===
I0000 ... session_basic.cc:486] RunDecodeSync
reply:  Paris.

=== Generate #2: log level = Error ===
reply:  Paris.

=== Generate #3: log level = Quiet ===
reply:  Paris.
```

The `Error → Quiet` transition is invisible here because the engine
emits no error-severity output on the happy path. The transition
becomes visible when a Generate call actually fails.

## Notes

- `SetMinLogLevel` is process-global. Calling it from one goroutine
  affects every Client in the process.
- The level applies to the C-side LiteRT-LM logger, not to Go's
  `log` / `slog` packages.
