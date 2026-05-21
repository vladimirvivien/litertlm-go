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

Each attempt runs two paths in sequence: the synthesized tool-call
path first, then the prompt-engineered fallback when the first path
does not deliver a value.

### Primary: synthesized tool-call capture

1. **Synthesize a capture tool** keyed on `reflect.TypeFor[T]()`. The
   tool's JSON-Schema parameters are derived from `T`'s exported
   fields (same reflection rules as `RegisterTool`). The tool is
   cached on the Client; subsequent calls of the same type reuse the
   registration.
2. **Open a one-hop Chat** with the synthesized tool attached and the
   prompt augmented with a directive instructing the model to deliver
   the structured value as the tool's arguments.
3. **Dispatch.** The model emits a tool call inside the family's
   tool-call markers (`<|tool_call|>...<|/tool_call|>` for Gemma 4,
   `<tool_call>...</tool_call>` for Qwen 3; per-family marker syntax
   lives in upstream `runtime/conversation/model_data_processor/`). The
   C-side `model_data_processor` extracts the JSON envelope under strict
   parsing and surfaces `{name, arguments}`. The wrapper unmarshals
   `arguments` into a fresh `T` and writes it to a per-call slot in
   `ctx.Value`.
4. **Return** the captured `*T`.

Markdown fences, prose preambles, and unbalanced braces cannot appear
in `arguments` — the family markers bracket the model's output and
the C-side parser is strict.

### Fallback: prompt-engineered + tolerant parse

The fallback runs when the primary path returns nil: `T` is not a
struct, chat construction failed, the transport errored, or the model
declined to call the tool.

1. **Schema reflection.** Walk `reflect.TypeFor[T]()` and emit a
   compact shape hint:

   ```
   {"title": <string>, "ingredients": [<string>], "steps": [<string>]}
   ```

   `json` tags including `,omitempty` and `-` are honored. Unsupported
   kinds (channel, func, non-string-keyed map) fail at call time.

2. **Prompt augmentation.** Prepend a default instruction containing
   the shape hint to the last text Part:

   ```
   Respond with valid JSON only — no commentary, no markdown
   fences. The output must match this shape:
   {shape}
   ```

   Override with `WithSchemaInstruction(s)` where `s` is a Printf
   format string containing one `%s` placeholder for the shape.

3. **Generate.** Routes through `Client.Generate` /
   `Client.GenerateMulti`; sampler params, `WithMaxOutputTokens`, and
   `ctx` cancellation apply.

4. **Tolerant extraction.** Strip markdown fences, locate the first
   balanced `{...}` (or `[...]` for slice `T`), respect string
   literals so braces inside string values do not confuse the depth
   counter.

5. **Unmarshal** into `*T` via `encoding/json`.

## Retry semantics

`WithRetries(n)` controls iteration count: each attempt runs the
primary path then the fallback path in sequence. The full pair counts
as one attempt; up to `1+n` attempts run before returning the last
parse error.

Retries fire on parse-path failures. Generate-phase errors (`ctx.Err()`,
FFI failures, model crashes) propagate immediately.

## Options

| Option                     | Effect                                                                                |
|----------------------------|---------------------------------------------------------------------------------------|
| `WithRetries(n)`           | Max retry attempts on parse failure. Default 0 (one total attempt).                   |
| `WithSchemaInstruction(s)` | Override the fallback path's default preamble. `s` is a Printf format string with one `%s`. |
| `WithMaxOutputTokens(n)`   | Cap output tokens on the fallback path.                                               |
| `WithSampler(p)`           | Override the Client's default sampler on the fallback path.                           |

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

## See also

- [`examples/structured/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/structured)
  — text-only Recipe extraction.
- [`examples/extract/`](https://github.com/vladimirvivien/litertlm-go/tree/main/examples/extract)
  — image-to-JSON extraction with `GenerateDataMulti`.
- [Client](client.md#multimodal-inputs) — the underlying multimodal API.
- [Tools](tools.md) — `RegisterTool` / `WithTool` for user-defined
  tool dispatch (the same machinery the primary path uses).
