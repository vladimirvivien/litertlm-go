# Runtime Metal Residency Control

This example enables Apple GPU `MTLResidencySet` tracking at initialization (`litertlm.WithMetalResidencySet(true)`), runs a generation turn, and dynamically toggles GPU weight residency at runtime via `client.SetMetalResidency`.

On Apple platforms (macOS and iOS) running the GPU backend, residency sets keep weights and allocations pinned in GPU memory to avoid paging overhead. On Linux, Windows, and CPU execution, the runtime ignores the setting safely without error.

## Running

```bash
go run ./examples/metal-residency \
  -get-lib v0.19.1 \
  -get-model litert-community/gemma3-1b-it-int4 \
  -backend gpu
```
