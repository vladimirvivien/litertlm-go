# chat — multi-turn Conversation API with JSON messages

Drives a chat-tuned model through the **Conversation API**, which applies
the model's chat template (e.g. Gemma's `<|turn>user … <turn|>`) on the C
side. Inputs and outputs are JSON-encoded — same format the underlying
LiteRT-LM CLI uses.

## What this example shows

- `Conversation` lifecycle: `NewEngineSettings` → `NewEngine` →
  `NewConversationConfig` → `engine.NewConversation` → `SendMessage` …
- Multi-turn chat: the conversation handle keeps prior turns in context so
  the second `SendMessage` is aware of the first.
- The JSON message shape:
  ```json
  {"role": "user", "content": "Hi, what is your name?"}
  ```
- Setting a system prompt via the `system_message_json` argument to
  `NewConversationConfig`.

## Prerequisites

1. Native shared library + `libGemmaModelConstraintProvider.*` staged in
   a directory pointed to by `LITERTLM_LIB`.
2. A **chat-tuned** `.litertlm` model file (Gemma instruct, Phi-4-instruct,
   Llama-Instruct, etc.). Base completion models will work too but waste
   the chat template.
3. Go 1.22+.

## Run

Two-turn built-in demo:

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/chat \
    -model /abs/path/to/gemma-4-E2B-it.litertlm
```

Single-turn one-shot mode:

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/chat \
    -model /abs/path/to/gemma-4-E2B-it.litertlm \
    -prompt "Explain TCP three-way handshake in two sentences."
```

| Flag        | Default                                   | Notes                                                          |
| ----------- | ----------------------------------------- | -------------------------------------------------------------- |
| `-model`    | (required)                                |                                                                |
| `-system`   | `"You are a friendly assistant."`         | Goes into the system message slot of the chat template.        |
| `-prompt`   | (empty → use the built-in two-turn demo)  | If set, sends one user message and exits.                      |
| `-backend`  | `"cpu"`                                   |                                                                |
| `-lib`      | `$LITERTLM_LIB`                           |                                                                |

## Expected output

```
user> Hi, what is your name?
bot>  {"role":"assistant","content":[{"type":"text","text":"Hi, my name is Gemma 4. I am a Large Language Model developed by Google DeepMind."}]}

user> Tell me a one-sentence fun fact about octopuses.
bot>  {"role":"assistant","content":[{"type":"text","text":"Octopuses have three hearts and blue blood, and they can change the color of their skin to blend in or startle their prey!"}]}
```

The reply is JSON; pull `.content[0].text` out with `encoding/json` if you
want just the prose.

## Notes

- The C API exposes the assistant reply as a JSON document — that's
  intentional, because chat-tuned models can also emit tool calls,
  multimodal segments, and reasoning channels (see Gemma 4's
  `<|channel>thought` blocks), all of which would need separate fields.
  This example just prints the JSON; in real code you'd unmarshal it.
- For **streaming** Conversation output token-by-token, swap
  `conv.SendMessage(...)` for `conv.SendMessageStreamCh(messageJSON, "")`
  — same channel idiom as the [`stream`](../stream/) example.
- The Conversation handle owns the dialogue history. Open a fresh
  `Conversation` for an unrelated topic, or reuse the same one for
  multi-turn context.
