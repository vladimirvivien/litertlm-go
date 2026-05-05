# Example: Typed Structured Output

This example shows `litertlm-go` can generate type-safe structured output from 
the model output. In the example, the code promps the model for `Recipe` 
(title + ingredients + steps), and it automatically parses
the response into a typed Go struct.

## What this example shows
- Create and configure a new instance of `litertlm.Client`.
- Uses `litertlm.GenerateData[T]` to generate a one-shot response that
  automatically populate a value of type `*T` from the model's JSON output.
- `litertlm.WithRetries(n)` — can retry if there are parse failures, in cases
  where the model emits improperly formatted JSON.
- `*GenerateDataError` — distinguishes parse failures (model returned
  text that couldn't unmarshal) from generate failures (FFI / ctx
  cancellation).

## Prerequisites

1. LiteRT-LM shared library files staged in`LITERTLM_LIB`.
2. A `.litertlm` model (i.e. Gemma 4). 
3. `litertlm-go`

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

- **Model reliability** E4B (4B parameters) follows the
  "JSON" instructions in most cases. The E2B Gemma model (the smallest model) 
  can be less consistent and may require retries.
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

