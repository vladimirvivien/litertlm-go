# Per-Turn Tool Choice (`examples/tool-choice`)

Demonstrates OpenAI-style `WithToolChoice` policies (`ToolChoiceAuto`,
`ToolChoiceNone`, `ToolChoiceRequired`, `ToolChoiceFunction`,
`ToolChoiceAllowedTools`) paired with `WithConstrainedDecoding(true)` and
registered Go tools.

Requires LiteRT-LM C API `1.1.0+` (`v0.20.0`). On C API `1.0.0` (`v0.19.1`)
runtimes, `WithToolChoice` returns `litertlm.ErrUnimplemented`.

## Run

```bash
go run ./examples/tool-choice \
    -lib /abs/path/to/litertlm-1.1.0-lib \
    -model /abs/path/to/model.litertlm
```
