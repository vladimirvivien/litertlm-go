# Structured output

`GenerateData[T]` and `GenerateDataMulti[T]` produce a typed `*T`
populated from the model's response, without manual JSON parsing in
caller code.

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

## Signatures

```go
func GenerateData[T any](
    ctx context.Context,
    c *Client,
    prompt string,
    opts ...RuntimeOption,
) (*T, error)

func GenerateDataMulti[T any](
    ctx context.Context,
    c *Client,
    parts []Part,
    opts ...RuntimeOption,
) (*T, error)
```

`T` must be a struct or pointer-to-struct for the primary path. Other
kinds (slices, scalars, maps) route through the fallback path described
below.

## How it works

GenerateData uses a two-tier strategy to extract structured data from the model:

### 1. Primary Path: Synthesized Tool Calling
* **Automatic Tool Creation**: The library reflects on your Go struct `T` and automatically registers a temporary "capture tool" representing its schema.
* **Model Directive**: It instructs the model to populate the fields of this tool to answer your prompt.
* **Direct Extraction**: When the model calls the tool, the library captures the arguments directly and unmarshals them into your struct `T`. This path leverages the model's native function-calling capabilities.

### 2. Fallback Path: Prompt Engineering & Tolerant JSON Parsing
If the model does not support tool calling, or if the tool call fails:
* **Schema Hinting**: The library generates a compact JSON shape hint from your Go struct.
* **Prompt Augmentation**: It instructs the model to return valid JSON matching that exact shape, with no markdown formatting or commentary.
* **Tolerant Parsing**: The library cleans up the model's output (stripping markdown fences, preambles, and conversational prose), extracts the raw JSON block, and unmarshals it into your struct `T`.

## Retry semantics

`WithRetries(n)` controls iteration count: each attempt runs the
primary path then the fallback path in sequence. The full pair counts
as one attempt; up to `1+n` attempts run before returning the last
parse error.

Retries fire on parse-path failures. Generate-phase errors (`ctx.Err()`,
FFI failures, model crashes) propagate immediately.

## Options

`GenerateData[T]` and `GenerateDataMulti[T]` accept any `RuntimeOption` (forwarding sampling, thinking, and penalty options to the underlying turns):

| Option                        | Effect                                                                                   |
|-------------------------------|------------------------------------------------------------------------------------------|
| `WithRetries(n)`              | Max retry attempts on parse failure. Default 0 (one total attempt).                      |
| `WithSchemaInstruction(s)`    | Override the fallback path's default preamble. `s` is a Printf format string with one `%s`. |
| `WithMaxOutputTokens(n)`      | Cap output tokens on the fallback path.                                                  |
| `WithSampler(p)`              | Override the Client's default sampler on the fallback path.                              |
| `WithTurnThinking(cfg)`       | Override reasoning behavior and token budget for the generation call.                    |
| `WithRepetitionPenalty(cfg)`  | Apply repetition/presence/frequency penalties during generation.                         |
| `WithNoRepeatNgram(cfg)`      | Ban repeating n-grams during generation.                                                 |
| `WithSuppressTokens(ids...)`  | Ban listed token IDs during generation.                                                  |

## Error handling

`GenerateData` returns a `*GenerateDataError` on failure. Use
`errors.As` to distinguish phases.

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

Type:

```go
type GenerateDataError struct {
    Phase    string  // "generate" | "parse"
    Err      error   // wrapped underlying error
    Raw      string  // model output that failed to parse (parse phase)
    Attempts int     // 1-indexed
}
```

## Multimodal: `GenerateDataMulti[T]`

`GenerateDataMulti[T]` accepts a `[]Part` carrying image, audio, and
text segments. Requires a multimodal model.

```go
img, err := litertlm.ImageFromFile("/path/to/recipe-card.jpg")
if err != nil { return err }

recipe, err := litertlm.GenerateDataMulti[Recipe](ctx, client,
    []litertlm.Part{
        img,
        litertlm.Text("Extract the recipe shown in the image."),
    },
    litertlm.WithRetries(2),
)
```

Placement rule for the tool-call directive (primary path) and the
schema instruction (fallback path): both prepend to the LAST text Part
in `parts`. When `parts` contains no text Part, a synthesized
`Text(...)` is appended at the end. The caller's slice is never
mutated.

`GenerateData[T]` is the text-only convenience over
`GenerateDataMulti[T]`.

## Grammar-constrained JSON (`WithConstraint`)

When working inside a `Chat` or `Client.Generate` call with a model that embeds constraint-provider metadata (such as Gemma 3 1B, Gemma 4 E2B, or Qwen 3 0.6B), you can enforce exact JSON Schema or regular-expression syntax at decode time instead of relying on post-generation parsing:

```go
chat, err := client.NewChat(ctx,
    litertlm.WithConstrainedDecoding(true),
    litertlm.WithConstraintProvider(litertlm.ConstraintProviderLLGuidance),
)
if err != nil {
    return err
}
defer chat.Close()

reply, err := chat.Send(ctx, "Extract Ada Lovelace's name and age as JSON.",
    litertlm.WithConstraint(litertlm.ConstraintJSONSchema, `{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}`),
)
```

Models without embedded constraint-provider metadata fail `NewChat` with `Failed to create GemmaModelConstraintProvider`. For portable struct extraction across any model, use `GenerateData[T]`. See [`examples/constrained-json/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/constrained-json).

## See also

- [Tutorial: Vision Q&A & Structured JSON Extraction](tutorial-vision-json.md) — walk through `GenerateDataMulti[T]` and grammar-constrained `WithConstraint`.
- [Examples Gallery](examples.md) — all 48 runnable repository examples.
- [`examples/structured/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/structured)
  — text-only Recipe extraction.
- [`examples/extract/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/extract)
  — image-to-JSON extraction with `GenerateDataMulti`.
- [`examples/constrained-json/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/constrained-json)
  — grammar-constrained JSON Schema and regex decoding.
- [Client](client.md#multimodal-inputs) — the underlying multimodal API.
- [Tools](tools.md) — `RegisterTool` / `WithTool` for user-defined
  tool dispatch (the same machinery the primary path uses).
