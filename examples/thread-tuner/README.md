# CPU Thread Tuner (`examples/thread-tuner`)

This example benchmarks prompt generation across different CPU thread allocations using `litertlm.WithNumThreads` to measure CPU scaling performance.

## How It Works

1. Iterates over thread counts (e.g. 1, 2, 4).
2. Initializes a new `litertlm.Client` for each configuration with `litertlm.WithNumThreads(n)`.
3. Measures generation latency and tokens per second for each thread count.

## Usage

```bash
go run . -model /path/to/model.litertlm
```

With automatic library and model provisioning:
```bash
go run . -get-lib v0.16.0 -get-model litert-community/gemma3-1b-it-int4
```
