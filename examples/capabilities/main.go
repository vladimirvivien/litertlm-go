// capabilities demonstrates zero-allocation model inspection using
// litertlm.InspectModel to query metadata, supported modalities, default
// sampling parameters, and hardware features from a .litertlm file.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	model := flag.String("model", "", "path to a .litertlm model file (required, or set LITERTLM_MODEL)")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.19.1)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	asJSON := flag.Bool("json", false, "output capabilities formatted as JSON")
	flag.Parse()

	ctx := context.Background()

	resolvedLib := *libPath
	if *getLib != "" {
		staged, err := litertlm.FetchLib(ctx, "", "", *getLib)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fetch library: %v\n", err)
			os.Exit(1)
		}
		resolvedLib = staged
	}

	resolvedModel := *model
	if resolvedModel == "" {
		resolvedModel = os.Getenv("LITERTLM_MODEL")
	}
	if *getModel != "" {
		staged, err := litertlm.FetchModel(ctx, *getModel)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fetch model: %v\n", err)
			os.Exit(1)
		}
		resolvedModel = staged
	}

	if resolvedModel == "" {
		fmt.Fprintln(os.Stderr, "--model or --get-model (or LITERTLM_MODEL env) is required")
		os.Exit(2)
	}

	if resolvedLib != "" {
		if err := litertlm.Load(resolvedLib, *backend, ""); err != nil {
			fmt.Fprintf(os.Stderr, "load library: %v\n", err)
			os.Exit(1)
		}
	}

	caps, err := litertlm.InspectModel(resolvedModel)
	if err != nil {
		fmt.Fprintf(os.Stderr, "inspect model: %v\n", err)
		os.Exit(1)
	}

	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(caps); err != nil {
			fmt.Fprintf(os.Stderr, "encode json: %v\n", err)
			os.Exit(1)
		}
		return
	}

	fmt.Printf("Model Capabilities for: %s\n", caps.Path)
	fmt.Println("--------------------------------------------------")
	fmt.Printf("  Model Type           : %s\n", caps.ModelType)
	if caps.MinRuntimeVersion != "" {
		fmt.Printf("  Min Runtime Version  : %s\n", caps.MinRuntimeVersion)
	}
	fmt.Printf("  Supported Modalities : %v\n", formatModalities(caps.SupportedModalities))
	if caps.MaxContextTokens > 0 {
		fmt.Printf("  Max Context Tokens   : %d (dynamic=%v)\n", caps.MaxContextTokens, caps.IsDynamicContext)
	} else {
		fmt.Printf("  Dynamic Context      : %v\n", caps.IsDynamicContext)
	}
	fmt.Printf("  Speculative Decoding : %v\n", caps.SupportsSpeculativeDecoding)
	fmt.Printf("  Thinking / Reasoning : %v\n", caps.SupportsThinking)
	fmt.Printf("  Function Calling     : %v\n", caps.SupportsFunctionCalling)
	fmt.Printf("  Default Sampler      : %s\n", formatSampler(caps.DefaultSampler))
	fmt.Printf("  Default Temperature  : %.2f\n", caps.DefaultTemperature)
	fmt.Printf("  Default Top-K        : %d\n", caps.DefaultTopK)
	fmt.Printf("  Default Top-P        : %.2f\n", caps.DefaultTopP)
	if caps.MaxVisionTokenBudget >= 0 {
		fmt.Printf("  Vision Token Budget  : %d tokens\n", caps.MaxVisionTokenBudget)
	} else {
		fmt.Printf("  Vision Token Budget  : N/A\n")
	}
	if len(caps.VisionSignatureLengths) > 0 {
		fmt.Printf("  Vision Signatures    : %v\n", caps.VisionSignatureLengths)
	}
	if caps.EmbeddingDimension > 0 {
		fmt.Printf("  Embedding Dimension  : %d\n", caps.EmbeddingDimension)
	}
	if len(caps.EmbeddingSignatureLengths) > 0 {
		fmt.Printf("  Embedding Signatures : %v\n", caps.EmbeddingSignatureLengths)
	}
	for _, m := range caps.SupportedModalities {
		backends := formatBackends(caps.ModalityBackends[m])
		brand := caps.ModalityNPUBrand[m]
		soc := caps.ModalitySoCName[m]
		if len(backends) > 0 || brand != litertlm.NPUBrandUnknown || soc != "" {
			fmt.Printf("  Modality %-11s : backends=%v", m.String(), backends)
			if brand != litertlm.NPUBrandUnknown {
				fmt.Printf(" npu=%s", brand.String())
			}
			if soc != "" {
				fmt.Printf(" soc=%s", soc)
			}
			fmt.Println()
		}
	}
}

func formatModalities(mods []litertlm.Modality) []string {
	res := make([]string, len(mods))
	for i, m := range mods {
		res[i] = m.String()
	}
	return res
}

func formatBackends(backends []litertlm.BackendType) []string {
	res := make([]string, len(backends))
	for i, b := range backends {
		res[i] = b.String()
	}
	return res
}

func formatSampler(s litertlm.SamplerType) string {
	switch s {
	case litertlm.SamplerTopK:
		return "TopK"
	case litertlm.SamplerTopP:
		return "TopP"
	case litertlm.SamplerGreedy:
		return "Greedy"
	case litertlm.SamplerTypeUnspecified:
		return "Unspecified"
	default:
		return fmt.Sprintf("SamplerType(%d)", s)
	}
}
