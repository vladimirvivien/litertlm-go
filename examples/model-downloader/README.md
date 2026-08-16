# model-downloader Example

Demonstrates automated downloading and running of `.litertlm` models from Hugging Face or direct URLs using `litertlm.FetchModel`.

---

## Features

* **Hugging Face Hub Resolution:** Resolves model repositories automatically (e.g. `litert-community/gemma3-1b-it-int4`).
* **Terminal Progress Bar:** Displays real-time download percentages, transfer rates, and progress bars.
* **Resumable Downloads:** Automatically resumes interrupted downloads using HTTP Range requests.
* **Immediate Inference:** Seamlessly initializes a LiteRT-LM `Client` with the staged model artifact.

---

## Usage

```bash
# Run with default Gemma 3 1B model
go run main.go

# Specify a custom model repository or direct URL
go run main.go -model litert-community/gemma-4-E2B-it

# Provide Hugging Face token for gated models
go run main.go -model google/gemma-3-4b-it -token hf_xxxxxxxxxxxx
```
