# Example: Typed Structured Output

Generates a typed Go value (`Recipe { title, ingredients, steps }`)
directly from a model prompt via `GenerateData[T]`. The model invokes
a synthesized capture tool whose arguments are unmarshaled into the
target type; a prompt-engineered fallback runs when the model declines
to call the tool.

## What this example shows

- `litertlm.GenerateData[T](ctx, client, prompt)` returning a typed
  `*T` populated from the model's response.
- `litertlm.WithRetries(n)` controlling attempt count when the
  fallback path's tolerant JSON parser fails.
- `*GenerateDataError` distinguishing parse failures from generate
  failures (FFI / ctx cancellation).

Recipe extraction routes through the primary tool-call path on Gemma 4;
the fallback path activates only when the model declines to call the
synthesized capture tool. See [docs/structured-output.md](../../docs/structured-output.md)
for the full pipeline.

## Prerequisites

1. LiteRT-LM shared library files staged in `LITERTLM_LIB`.
2. A `.litertlm` model (e.g. Gemma 4).

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

- **Primary path.** Gemma 4 E2B and E4B both reliably invoke the
  synthesized capture tool on this prompt shape. The model may leave
  individual struct fields empty when the JSON-Schema `required`
  marker is not strictly honored; the wrapper unmarshals partial
  arguments and zero-values the missing fields.
- **Fallback path.** Activates when the model declines to call the
  tool. Augments the prompt with a JSON-shape instruction and runs a
  tolerant parser over the response. `WithSchemaInstruction(s)`
  overrides the fallback's default preamble; `s` is a Printf format
  string with one `%s` for the shape:

  ```go
  litertlm.WithSchemaInstruction(
      "Output strictly valid JSON matching: %s. Do not include any explanation.")
  ```

  The fallback's shape hint for the `Recipe` struct above:

  ```
  {"title": <string>, "ingredients": [<string>], "steps": [<string>]}
  ```

