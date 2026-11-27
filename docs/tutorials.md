# Tutorials

Step-by-step walkthroughs help you move from a blank `go.mod` file to running real local AI workflows inside your own Go binaries. Each tutorial focuses on one end-to-end capability—streaming inference, vision-to-JSON extraction, multimodal semantic search, and branching tool-calling agents—so you can learn each pattern with copy-ready Go code and runnable repository examples.

!!! tip "Before You Begin"

    Every tutorial needs only **Go 1.26 or newer**. Native LiteRT-LM shared libraries (`v0.19.1+`) and public `.litertlm` models are fetched and cached automatically on first run via `litertlm.FetchLib` and `litertlm.FetchModel`.

<div class="grid cards" markdown>

-   :material-rocket-launch-outline:{ .lg .middle } __1. Your First Local LLM App in 5 Minutes__

    ---

    Provision the native runtime and Gemma 3 1B model from Go, inspect model metadata without booting an engine, stream generated tokens, and read live throughput benchmarks.

    **Model:** `gemma3-1b-it-int4` · **Time:** ~5 min

    [:octicons-arrow-right-24: Start Tutorial 1](tutorial-first-app.md)

-   :material-eye-check-outline:{ .lg .middle } __2. Vision Q&A & Structured JSON Extraction__

    ---

    Feed images into multimodal inference with `GenerateMulti`, extract typed Go structs with `GenerateDataMulti[T]`, and lock syntax at decode time using JSON Schema and regex grammars.

    **Model:** Gemma 4 E2B / E4B · **Time:** ~10 min

    [:octicons-arrow-right-24: Start Tutorial 2](tutorial-vision-json.md)

-   :material-image-search-outline:{ .lg .middle } __3. Local Multimodal Semantic Search & Classification__

    ---

    Embed text and images into a single shared vector space with `EmbeddingClient`, index local directories, run cross-modal queries, classify zero-shot labels, and tune Matryoshka vector sizes.

    **Model:** EmbeddingGemma 2 (ungated) · **Time:** ~10 min

    [:octicons-arrow-right-24: Start Tutorial 3](tutorial-semantic-search.md)

-   :material-source-branch:{ .lg .middle } __4. Tool-Calling Agents & KV-Cache Branching__

    ---

    Turn Go functions into auto-dispatched tools with `RegisterTool`, add error-recovery and `ToolChoice` guardrails, and fork prefilled prompts cleanly using `Chat.Clone` and KV checkpoints.

    **Model:** `gemma3-1b-it-int4` / Gemma 4 · **Time:** ~12 min

    [:octicons-arrow-right-24: Start Tutorial 4](tutorial-tools-branching.md)

</div>

Looking for a focused single-feature snippet rather than a full walkthrough? Visit the [Examples Gallery](examples.md) to browse all 48 runnable programs under `examples/`.
