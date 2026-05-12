# Example: Typed Structured Output

Generates a typed Go value (`Recipe { title, ingredients, steps }`)
directly from a model prompt via `GenerateData[T]`, which appends a
JSON-shape instruction and unmarshals the response into the target
type.

## What this example shows

- `litertlm.GenerateData[T](ctx, client, prompt)` returning a `*T`
  populated from the model's JSON output.
- `litertlm.WithRetries(n)` to retry on parse failure when the model
  emits malformed JSON.
- `*GenerateDataError` distinguishing parse failures (model returned
  text that did not unmarshal) from generate failures (FFI / ctx
  cancellation).

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

- **Model reliability.** Gemma 4 E4B (4B parameters) follows the
  injected JSON instruction reliably; the smaller E2B variant is
  less consistent and benefits from `WithRetries`.
- **Customising the instruction** - use `WithSchemaInstruction(s)` to
  override the default preamble.

  ```go
  litertlm.WithSchemaInstruction(
      "Output strictly valid JSON matching: %s. Do not include any explanation.")
  ```

- **What does the shape hint look like?** For the `Recipe` struct
  above, the helper inserts:

  ```
  {"title": <string>, "ingredients": [<string>], "steps": [<string>]}
  ```

