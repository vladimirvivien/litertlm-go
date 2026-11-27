# Tool-Calling Agents & KV-Cache Branching

Turning a local model into a reliable assistant takes two primitives: letting the model invoke typed Go functions safely, and branching conversations cheaply without paying prompt prefill costs twice. In this tutorial, you build a multi-hop tool-calling agent with `RegisterTool`, add error-recovery policies, concurrency caps, and `ToolChoice` constraints, and fork prefilled conversation state using `Chat.Clone` and low-level KV checkpoints.

<div class="lrt-meta-bar">
  <span class="lrt-pill lrt-pill--coral">~12 min</span>
  <span class="lrt-pill lrt-pill--teal">Tools &amp; KV Cache</span>
  <span class="lrt-pill lrt-pill--gold">gemma3-1b-it-int4 / Gemma 4</span>
  <span class="lrt-pill lrt-pill--plum">High &amp; Low Level</span>
</div>

---

## Step 1 — Register Typed Go Tools with Auto-Dispatch

`litertlm.RegisterTool` uses Go reflection over your input struct to generate an OpenAI-style JSON Schema automatically. Field names come from `json` tags (or lowercased field names), non-pointer fields become required parameters, and `description` struct tags guide the model on what values to pass.

Attach registered tools to a `Chat` using `WithTool(...)`. When `chat.Send` runs, `litertlm-go` intercepts any tool calls emitted by the model, invokes your Go handler, feeds the serialized result back into the conversation, and returns the final natural-language answer—all inside a single `chat.Send` call:

```go
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"runtime"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

type WeatherIn struct {
	City string `json:"city" description:"City name to check weather for, e.g. Tokyo"`
}

type WeatherOut struct {
	City       string  `json:"city"`
	TempC      float64 `json:"temp_c"`
	Conditions string  `json:"conditions"`
}

var weatherDB = map[string]WeatherOut{
	"Tokyo":  {City: "Tokyo", TempC: 22, Conditions: "clear"},
	"Boston": {City: "Boston", TempC: 15, Conditions: "breezy"},
}

func main() {
	ctx := context.Background()
	libDir, err := litertlm.FetchLib(ctx, runtime.GOOS, runtime.GOARCH, "v0.19.1")
	if err != nil { log.Fatal(err) }
	modelPath, err := litertlm.FetchModel(ctx, "litert-community/gemma3-1b-it-int4")
	if err != nil { log.Fatal(err) }

	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	client, err := litertlm.New(ctx, litertlm.WithLib(libDir), litertlm.WithModel(modelPath))
	if err != nil { log.Fatal(err) }
	defer client.Close()

	lookupWeather, err := litertlm.RegisterTool(client, "lookup_weather",
		"Return current weather conditions for a city.",
		func(_ context.Context, in WeatherIn) (WeatherOut, error) {
			out, ok := weatherDB[in.City]
			if !ok {
				return WeatherOut{}, fmt.Errorf("no forecast found for %q", in.City)
			}
			return out, nil
		},
		// Inform-on-error lets the model recover if an unknown city is queried:
		litertlm.WithToolPolicy(litertlm.ToolPolicyInformOnError),
	)
	if err != nil { log.Fatal(err) }

	chat, err := client.NewChat(ctx,
		litertlm.WithSystemPrompt("You answer weather questions using lookup_weather. If a lookup fails, apologize or try a nearby city mentioned by the user."),
		litertlm.WithTool(lookupWeather),
		litertlm.WithMaxToolHops(4),
	)
	if err != nil { log.Fatal(err) }
	defer chat.Close()

	reply, err := chat.Send(ctx, "Check Atlantis weather; if unavailable, check Tokyo.")
	if err != nil { log.Fatal(err) }
	fmt.Println(reply.Text())
}
```

### Bounded Tool Loops & Error Recovery

* **`WithMaxToolHops(n)`** caps model-to-tool round-trips (default `5`) so an indecisive model cannot loop indefinitely. When exceeded, check `errors.Is(err, litertlm.ErrToolHopsExceeded)` or unwrap `*litertlm.ToolHopsError` for partial replies.
* **`ToolPolicyReturnOnError`** (the default) immediately aborts `chat.Send` and returns the Go error. Choose it for authorization failures or corrupted state.
* **`ToolPolicyInformOnError`** serializes `{"error": "..."}` back into the conversation so the model can apologize, adjust arguments, or fall back to another tool call.

---

## Step 2 — Run Concurrent Tool Batches & Enforce `ToolChoice`

When a model requests several independent tool calls in a single turn (for example, looking up two cities simultaneously), pass `litertlm.WithMaxConcurrentTools(4)` on `chat.Send`. Handlers execute concurrently in worker goroutines up to the specified limit, while results are stitched back into the conversation in the exact call order the model produced. Make sure concurrent handlers synchronize access to shared mutable state.

On LiteRT-LM C API `1.1.0+`, you can also compel or restrict tool usage per turn via `WithToolChoice` when opening a chat with `WithConstrainedDecoding(true)`:

```go
forcedChat, err := client.NewChat(ctx,
	litertlm.WithConstrainedDecoding(true),
	litertlm.WithTool(lookupWeather),
)
if err != nil { log.Fatal(err) }
defer forcedChat.Close()

reply, err := forcedChat.Send(ctx, "How is Boston today?",
	litertlm.WithMaxOutputTokens(64),
	litertlm.WithToolChoice(litertlm.ToolChoiceFunction("lookup_weather")),
)
if errors.Is(err, litertlm.ErrUnimplemented) {
	log.Println("WithToolChoice requires a LiteRT-LM C API 1.1+ shared library")
}
```

Available policies include `ToolChoiceAuto()`, `ToolChoiceNone()`, `ToolChoiceRequired()`, `ToolChoiceFunction(name)`, `ToolChoiceAllowedTools(mode, names...)`, and `ToolChoiceRawJSON(payload)`.

---

## Step 3 — Branch Prefilled Conversations with `Chat.Clone`

Prefilling long system prompts, style rules, or reference documents can take hundreds of milliseconds on edge CPUs. Rather than rebuilding a `Chat` from scratch for each parallel candidate or speculative follow-up, call `chat.Clone()` after prefilling shared context. Each clone starts with an independent KV cache pointing at identical history and must be closed cleanly while keeping the root `Chat` open:

```go
chat, err := client.NewChat(ctx,
	litertlm.WithSystemPrompt("You are a concise assistant. Answer in one short sentence."),
)
if err != nil { log.Fatal(err) }
defer chat.Close()

if _, err := chat.Send(ctx, "Remember: Comet is a tricolor corgi who loves fetching tennis balls."); err != nil {
	log.Fatal(err)
}

branchB, err := chat.Clone()
if err != nil {
	if errors.Is(err, litertlm.ErrUnimplemented) {
		log.Println("Loaded executor does not support Session.Clone yet")
		return
	}
	log.Fatal(err)
}
defer branchB.Close()

replyA, _ := chat.Send(ctx, "What breed is Comet?")
replyB, _ := branchB.Send(ctx, "Suggest a playful nickname for Comet.")
fmt.Println("Branch A:", replyA.Text())
fmt.Println("Branch B:", replyB.Text())
```

---

## Step 4 — Checkpoint & Rewind at the Low-Level `Session` Layer

When you need explicit byte-level prompt control (such as tree-of-thought search or reusable completion prefixes without chat-template wrapping), drop down to `engine.NewSession` with `SaveCheckpoint`, `RewindToCheckpoint`, `Session.Clone`, and `RunDecodeAsyncCh`:

```go
session, err := client.Engine().NewSession(0)
if err != nil { log.Fatal(err) }
defer session.Delete()

prefix, _ := litertlm.NewTextInputString("In a small seaside village,")
defer prefix.Delete()

_ = session.RunPrefill([]litertlm.InputData{prefix})
_ = session.SaveCheckpoint("after-prefix")

clonedSess, err := session.Clone()
if err != nil { log.Fatal(err) }
defer clonedSess.Delete()

// Decode Branch A on session, rewind KV cache, then decode Branch B cleanly:
inA, _ := litertlm.NewTextInputString(" an old lighthouse keeper")
defer inA.Delete()
_ = session.RunPrefill([]litertlm.InputData{inA})
for sc := range session.RunDecodeAsyncCh() {
	if sc.Err != nil { log.Fatal(sc.Err) }
	fmt.Print(sc.Text)
}
fmt.Println()

_ = session.RewindToCheckpoint("after-prefix")
```

---

## Try the Repository Examples

```bash
# Auto-dispatched calculator tool
go run ./examples/autotool -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4

# ToolPolicyReturnOnError vs ToolPolicyInformOnError
go run ./examples/tool-policy -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4 -policy inform

# Forced / allowed ToolChoice policies (C API 1.1+)
go run ./examples/tool-choice -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4

# High-level Chat.Clone and low-level Session checkpoints
go run ./examples/clone -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4
go run ./examples/checkpoints -get-lib v0.19.1 -get-model litert-community/gemma3-1b-it-int4
```

## Next Steps

* Explore schema reflection rules and manual `NewRawTool` dispatch in the [Tools Guide](tools.md).
* Review transcript seeding (`WithInitialMessages`) and KV-cache token tracking (`Chat.TokenCount`) in the [Chat Guide](chat.md).
* Inspect every handle lifetime rule in the [Low-Level API](low-level.md).
