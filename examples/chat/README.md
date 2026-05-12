# Example: Multi-Turn Conversation with the Chat API

Runs a multi-turn conversation against the `Chat` API.

## What this example shows

- `Client.NewChat(ctx, WithSystemPrompt(...))` to open a chat with a
  system message.
- `Chat.Send(ctx, message)` returning a `*Reply` whose `Text()` gives
  the assistant's response.
- Successive `Send` calls automatically carry prior turns in the
  conversation's KV cache.

## Prerequisites

1. LiteRT-LM shared library files staged in `LITERTLM_LIB`.
2. A `.litertlm` chat-tuned model (e.g. Gemma 4).

## Run

```bash
LITERTLM_LIB=/abs/path/to/dist/lib \
    go run ./examples/chat \
    -model /abs/path/to/gemma-4-E2B-it.litertlm
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
bot>  Hi, my name is Gemma 4. I am a Large Language Model developed by Google DeepMind.

user> Tell me a one-sentence fun fact about octopuses.
bot>  Octopuses have three hearts and blue blood, and they can change the color of their skin to blend in or startle their prey!
```
