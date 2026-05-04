# Structured output

Use `GenerateData[T]` to return an instance of `*T` that is automatically
unmarshaled with the model's JSON output. For this convenience to work properly,
you must first declare your data type as shown in the snippet below.

```go
type Recipe struct {
    Title       string   `json:"title"`
    Ingredients []string `json:"ingredients"`
    Steps       []string `json:"steps"`
}

recipe, err := litertlm.GenerateData[Recipe](ctx, client,
    "Recipe for chocolate chip cookies.",
    litertlm.WithRetries(2),
)
if err != nil {
    return err
}
fmt.Println(recipe.Title)
fmt.Println(recipe.Ingredients)
```

## Using GenerateData

`Client.GenerateData` mirrors the functionalities of `Client.Generate(ctx, prompt, opts...)` 
with the additional type parameter to store the response from the LLM.


```go
func GenerateData[T any](
    ctx context.Context,
    c *Client,
    prompt string,
    opts ...GenOption,
) (*T, error)
```


### How it works

1. **Schema reflection.** Recursively walk `reflect.TypeFor[T]()` and
   emit a compact instruction-friendly hint:

    ```
    {"title": <string>, "ingredients": [<string>], "steps": [<string>]}
    ```

    Honors `json` tags including `,omitempty` and `-`. Unsupported kinds (channel, func,
    non-string-keyed map) fail at call time, before hitting the model.

2. **Prompt augmentation.** A special markup  hint is inserted before the
   user prompt with a default instruction:

   ```
   Respond with valid JSON only — no commentary, no markdown
   fences. The output must match this shape:
   {shape}
   ```

    You can use `WithSchemaInstruction(s)` to overide the instruction
    if you want different wording where string `s` must contain one 
    `%s` placeholder.

3. **Generate.** Internally `Client.GenerateData` uses `Client.Generate`, so
   `ctx` cancellation, sampler params, and `WithMaxOutputTokens` all
   work as usual.

4. **Tolerant extraction.** The model often wraps JSON in
   ```` ```json … ``` ``` fences or a prose preamble. The extractor
   strips fences, finds the first balanced `{...}` (or `[...]` for
   slice `T`), and respects string literals so braces inside string
   values don't fool the depth counter.

5. **Unmarshal** Finally, the returned JSON is unmarshaled into a value of type 
   `*T` via `encoding/json`.

If extraction or unmarshal fails, the error wraps in
`*GenerateDataError` and — if `WithRetries(n)` was set — the call
re-runs. Up to `1 + n` total attempts.

### Function options

| Option                              | Effect                                                                  |
|-------------------------------------|-------------------------------------------------------------------------|
| `WithRetries(n)`                    | Max retry attempts on parse failure. Default 0 (one total attempt).     |
| `WithSchemaInstruction(s)`          | Override the default preamble. `s` is a Printf format string with one `%s` for the shape. |
| `WithMaxOutputTokens(n)`            | Cap output tokens.                                                      |
| `WithSampler(p)`                    | Override the Client's default sampler.                                  |

### Error handling
`GenerateData` returns a structured error to allow users to distinguish between structural
and parsing errors. Parsing errors can happen if the model sends unfinished data.

```go
recipe, err := litertlm.GenerateData[Recipe](ctx, client, prompt)
if err != nil {
    var gd *litertlm.GenerateDataError
    if errors.As(err, &gd) {
        switch gd.Phase {
        case "parse":
            log.Printf("model returned bad JSON after %d attempts:\n%s",
                gd.Attempts, gd.Raw)
        case "generate":
            log.Printf("model call itself failed: %v", gd.Err)
        }
    }
    return err
}
```

Type `*GenerateDataError`:

```go
type GenerateDataError struct {
    Phase    string  // "generate" | "parse"
    Err      error   // wrapped underlying error
    Raw      string  // model output that failed to parse (parse phase)
    Attempts int     // 1-indexed
}
```

**Retries fire only on parse failures.** Generate-phase errors
(`ctx.Err()`, FFI failures, model crashes) propagate the error and fails immediately.

## See also

- [`examples/structured/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/structured)
  — full Recipe extraction demo.
- [Client](client.md) — the underlying `Generate` path.
