# structured — type-safe structured-output extraction

Asks the model for a `Recipe` (title + ingredients + steps), parses
the response into the typed Go struct, and pretty-prints it. Uses
`litertlm.GenerateData[Recipe]` under the hood — Tier A
implementation: prompt-engineered JSON instruction + tolerant parse +
optional retry on failure.

## What this example shows

- `litertlm.GenerateData[T]` — generic helper that returns `*T`
  populated from the model's JSON output.
- `WithRetries(n)` — retry loop for parse failures. Local LLMs
  occasionally emit invalid JSON; bumping retries usually fixes it.
- `*GenerateDataError` — distinguishes parse failures (model returned
  text that couldn't unmarshal) from generate failures (FFI / ctx
  cancellation).
- The shape hint is auto-derived via reflection from the `Recipe`
  struct's exported fields and `json` tags.

## Prerequisites

1. Native shared library + `libGemmaModelConstraintProvider.*` staged
   in `$LITERTLM_LIB`.
2. A chat-tuned `.litertlm` model. E4B is recommended for reliability;
   E2B works but may need higher `-retries`.
3. Go 1.26+.

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/structured \
    -model /abs/path/to/gemma-4-E4B-it.litertlm
```

Override the prompt or retry count:

```bash
go run ./examples/structured -model … -prompt "Quick spaghetti carbonara." -retries 3
```

| Flag       | Default                                    | Notes                                    |
| ---------- | ------------------------------------------ | ---------------------------------------- |
| `-model`   | (required)                                 | Path to a `.litertlm` model.             |
| `-prompt`  | `"Recipe for chocolate chip cookies."`     | Free-text description of what to extract. |
| `-retries` | `2`                                        | Max retry attempts on parse failure.     |
| `-backend` | `"cpu"`                                    |                                          |
| `-lib`     | `$LITERTLM_LIB`                            |                                          |

## Expected output

A JSON-pretty-printed Recipe struct, e.g.:

```json
{
  "title": "Chocolate Chip Cookies",
  "ingredients": [
    "2 1/4 cups all-purpose flour",
    "1 tsp baking soda",
    "1 tsp salt",
    "1 cup butter, softened",
    "..."
  ],
  "steps": [
    "Preheat oven to 375°F.",
    "Combine dry ingredients.",
    "..."
  ]
}
```

## Notes

- **Reliability is model-dependent.** E4B (4B parameters) follows the
  "JSON only, no fences" instruction reliably on the first try in
  most cases. E2B (2B parameters) is less consistent — start with
  `-retries 3` or higher.
- **Tier A is best-effort.** The prompt-injection approach has no
  hard guarantee that the model emits valid JSON. The proper fix is
  constrained decoding driven by a JSON schema, which requires the
  C-API hook upstream LiteRT-LM does not yet expose. When that lands,
  `GenerateData[T]` swaps to Tier B internally with no caller
  changes.
- **Customising the instruction.** `WithSchemaInstruction(s)` lets
  you override the default preamble. `s` must be a `fmt.Sprintf`
  format string with one `%s` placeholder where the shape hint goes:

  ```go
  litertlm.WithSchemaInstruction(
      "Output strictly valid JSON matching: %s. Do not include any explanation.")
  ```

- **What does the shape hint look like?** For the `Recipe` struct
  above, the helper inserts:

  ```
  {"title": <string>, "ingredients": [<string>], "steps": [<string>]}
  ```

  Models follow this short form more reliably than full JSON Schema.
- **Top-level slices** are supported: `GenerateData[[]Recipe]` asks
  the model for a JSON array. The shape hint becomes
  `[{"title": <string>, ...}]`.
