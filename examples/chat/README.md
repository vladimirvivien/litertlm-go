# Example: Multi-Turn Conversation with the Chat API

This examples demonstrate the  **Chat API** `Send` and `Reply`
multi-turn conversation.

## What this example shows

- Use a new of `litertlmgo.Client` to create new `Chat`
- Use `Client.Send` to send message to LLM engine.
- Use the `Reply` to inspect message response.
- The API automatically keeps prior turns in context.


## Prerequisites

1. LiteRT-LM shared library files staged in`LITERTLM_LIB`.
2. A `.litertlm` model (i.e. Gemma 4). 
3. `litertlm-go`

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
