# Structured output

`GenerateData[T]` returns a populated `*T` from the model's JSON
output. The helper handles three things you'd otherwise build
yourself: deriving an instruction from `T`, extracting JSON out of
free-form model text, and unmarshaling.

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

## Signature

```go
func GenerateData[T any](
    ctx context.Context,
    c *Client,
    prompt string,
    opts ...GenOption,
) (*T, error)
```

Methods can't introduce new type parameters in Go, so this is a
package-level function. Same shape as
`Client.Generate(ctx, prompt, opts...)` with a generic type param up
front and an extra client argument.

## How it works (Tier A)

1. **Schema reflection.** Recursively walk `reflect.TypeFor[T]()` and
   emit a compact instruction-friendly hint:

    ```
    {"title": <string>, "ingredients": [<string>], "steps": [<string>]}
    ```

    Honors `json` tags including `,omitempty` and `-`. Top-level
    slices/arrays produce `[...]`. Unsupported kinds (channel, func,
    non-string-keyed map) fail at call time, before hitting the model.

2. **Prompt augmentation.** The shape hint is inserted before the
   user prompt with a default instruction:

    > Respond with valid JSON only — no commentary, no markdown
    > fences. The output must match this shape:
    > `{shape}`

    Override via `WithSchemaInstruction(s)` if you want different
    wording; `s` must contain one `%s` placeholder.

3. **Generate.** Routes through the same `Client.Generate` path, so
   `ctx` cancellation, sampler params, and `WithMaxOutputTokens` all
   work as usual.

4. **Tolerant extraction.** The model often wraps JSON in
   ```` ```json … ``` ``` fences or a prose preamble. The extractor
   strips fences, finds the first balanced `{...}` (or `[...]` for
   slice `T`), and respects string literals so braces inside string
   values don't fool the depth counter.

5. **Unmarshal** into `*T` via `encoding/json`.

If extraction or unmarshal fails, the error wraps in
`*GenerateDataError` and — if `WithRetries(n)` was set — the call
re-runs. Up to `1 + n` total attempts.

## Per-call options

| Option                              | Effect                                                                  |
|-------------------------------------|-------------------------------------------------------------------------|
| `WithRetries(n)`                    | Max retry attempts on parse failure. Default 0 (one total attempt).     |
| `WithSchemaInstruction(s)`          | Override the default preamble. `s` is a Printf format string with one `%s` for the shape. |
| `WithMaxOutputTokens(n)`            | Cap output tokens.                                                      |
| `WithSampler(p)`                    | Override the Client's default sampler.                                  |

## Errors

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

`*GenerateDataError`:

```go
type GenerateDataError struct {
    Phase    string  // "generate" | "parse"
    Err      error   // wrapped underlying error
    Raw      string  // model output that failed to parse (parse phase)
    Attempts int     // 1-indexed
}
```

Implements `Error()` and `Unwrap()` so `errors.Is` and `errors.As`
reach the inner error.

**Retries fire only on parse failures.** Generate-phase errors
(`ctx.Err()`, FFI failures, model crashes) propagate immediately.

## Top-level slices

`GenerateData[[]Recipe]` works — the shape hint becomes
`[{...}]` and the extractor finds the first balanced `[...]`.

```go
recipes, err := litertlm.GenerateData[[]Recipe](ctx, client,
    "Three recipes for breakfast.",
    litertlm.WithRetries(2),
)
// recipes is *[]Recipe
for _, r := range *recipes {
    fmt.Println(r.Title)
}
```

## Reliability

Tier A is best-effort. Local LLMs occasionally emit invalid JSON
despite the instruction:

- **E4B** (4B parameters, instruction-tuned): one-shot success in the
  large majority of cases. `WithRetries(2)` handles the long tail.
- **E2B** (2B parameters): more variance. Start at `WithRetries(3)`
  or higher. Smaller models sometimes wrap the JSON in markdown
  fences or add prose; the extractor handles both, but malformed
  JSON contents are an unrecoverable parse error per attempt.

## Roadmap: Tier B

The proper fix for unreliable structured output is **constrained
decoding**: the engine masks logits at each token-generation step so
the model can only emit tokens that keep the partial output valid
against a JSON schema.

LiteRT-LM has the underlying plumbing (LlGuidance integration in
`runtime/components/constrained_decoding/`), but the public C API
exposes only the boolean toggle, not a schema-delivery hook. Tier B
is gated on that upstream addition.

When Tier B lands, `GenerateData[T]` swaps internals to use it. The
public Go signature stays the same — your callers won't see a
breaking change.

## See also

- [`examples/structured/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/structured)
  — full Recipe extraction demo.
- [Client](client.md) — the underlying `Generate` path.
