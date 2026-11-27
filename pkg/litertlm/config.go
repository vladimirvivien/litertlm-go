package litertlm

import (
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"
)

type configFile struct {
	Default *jsonModelConfig            `json:"default,omitempty"`
	Models  map[string]*jsonModelConfig `json:"models,omitempty"`
}

type jsonModelConfig struct {
	Backend                     *string  `json:"backend,omitempty"`
	MaxTokens                   *int     `json:"max_tokens,omitempty"`
	CacheDir                    *string  `json:"cache_dir,omitempty"`
	SpeculativeDecodingEnabled  *bool    `json:"speculative_decoding_enabled,omitempty"`
	SpeculativeDecodingMode     *string  `json:"speculative_decoding_mode,omitempty"`
	SpeculativeDecodingLenience *float32 `json:"speculative_decoding_lenience,omitempty"`
	ParallelSectionLoading      *bool    `json:"parallel_section_loading,omitempty"`
	EnableYNNPACK               *bool    `json:"enable_ynnpack,omitempty"`
	MaxVisionTokensPerImage     *int     `json:"max_vision_tokens_per_image,omitempty"`
	MetalResidencySet           *bool    `json:"metal_residency_set,omitempty"`
	GPUDecodeStepsPerSync       *int     `json:"gpu_decode_steps_per_sync,omitempty"`
	GPUWaitForWeightUploads     *bool    `json:"gpu_wait_for_weight_uploads,omitempty"`
	UseRingbuffersLocalAttn     *bool    `json:"use_ringbuffers_local_attention,omitempty"`
	SingleThreadedExecution     *bool    `json:"single_threaded_execution,omitempty"`
	EnableDebugger              *bool    `json:"enable_debugger,omitempty"`
	SelectedSignatures          []string `json:"selected_signatures,omitempty"`
	NumThreads                  *int     `json:"num_threads,omitempty"`
	AudioNumThreads             *int     `json:"audio_num_threads,omitempty"`
	VisionDispatchLibDir        *string  `json:"vision_dispatch_lib_dir,omitempty"`
	AudioDispatchLibDir         *string  `json:"audio_dispatch_lib_dir,omitempty"`
	MaxInputLength              *int     `json:"max_input_length,omitempty"`
	MinInputLength              *int     `json:"min_input_length,omitempty"`
	LoraRank                    *int     `json:"lora_rank,omitempty"`
	SupportedLoraRanks          []int    `json:"supported_lora_ranks,omitempty"`
	AudioLoraRank               *int     `json:"audio_lora_rank,omitempty"`
	SupportedAudioLoraRanks     []int    `json:"supported_audio_lora_ranks,omitempty"`
	SamplerType                 *string  `json:"sampler_type,omitempty"`
	Temperature                 *float32 `json:"temperature,omitempty"`
	TopK                        *int     `json:"top_k,omitempty"`
	TopP                        *float32 `json:"top_p,omitempty"`
	Seed                        *int     `json:"seed,omitempty"`
	LibPath                     *string  `json:"lib_path,omitempty"`
	LibName                     *string  `json:"lib_name,omitempty"`
	ModelPath                   *string  `json:"model_path,omitempty"`
}

// WithConfigFile reads settings from a LiteRT-LM config.json file and applies them
// to the client configuration.
func WithConfigFile(path, modelID string) Option {
	return func(c *clientConfig) {
		if c.err != nil {
			return
		}
		data, err := os.ReadFile(path)
		if err != nil {
			c.err = fmt.Errorf("read config file: %w", err)
			return
		}

		var cf configFile
		if err = json.Unmarshal(data, &cf); err != nil {
			c.err = fmt.Errorf("parse config file: %w", err)
			return
		}

		if cf.Default != nil {
			if err = c.applyModelConfig(cf.Default); err != nil {
				c.err = err
				return
			}
		}

		if modelID != "" && cf.Models != nil {
			if mc, ok := cf.Models[modelID]; ok {
				if err = c.applyModelConfig(mc); err != nil {
					c.err = err
					return
				}
			}
		}
	}
}

func (c *clientConfig) applyModelConfig(mc *jsonModelConfig) error {
	if mc == nil {
		return nil
	}
	if mc.Backend != nil {
		c.backend = *mc.Backend
	}
	if mc.MaxTokens != nil {
		c.maxTokens = *mc.MaxTokens
	}
	if mc.CacheDir != nil {
		c.cacheDir = *mc.CacheDir
	}
	if mc.SpeculativeDecodingEnabled != nil {
		c.speculativeDecodingEnabled = new(*mc.SpeculativeDecodingEnabled)
	}
	if mc.SpeculativeDecodingMode != nil || mc.SpeculativeDecodingLenience != nil {
		var sd SpeculativeDecodingConfig
		if c.speculativeDecoding != nil {
			sd = *c.speculativeDecoding
		}
		if mc.SpeculativeDecodingMode != nil {
			mode, err := parseSpeculativeDecodingMode(*mc.SpeculativeDecodingMode)
			if err != nil {
				return err
			}
			sd.Mode = mode
		}
		if mc.SpeculativeDecodingLenience != nil {
			sd.Lenience = *mc.SpeculativeDecodingLenience
		}
		c.speculativeDecoding = new(sd)
	}
	if mc.ParallelSectionLoading != nil {
		c.parallelSectionLoading = new(*mc.ParallelSectionLoading)
	}
	if mc.EnableYNNPACK != nil {
		c.enableYNNPACK = new(*mc.EnableYNNPACK)
	}
	if mc.MaxVisionTokensPerImage != nil {
		c.maxVisionTokensPerImage = new(*mc.MaxVisionTokensPerImage)
	}
	if mc.MetalResidencySet != nil {
		c.metalResidencySet = new(*mc.MetalResidencySet)
	}
	if mc.GPUDecodeStepsPerSync != nil {
		c.gpuDecodeStepsPerSync = new(*mc.GPUDecodeStepsPerSync)
	}
	if mc.GPUWaitForWeightUploads != nil {
		c.gpuWaitForWeightUploads = new(*mc.GPUWaitForWeightUploads)
	}
	if mc.UseRingbuffersLocalAttn != nil {
		c.useRingbuffersLocalAttn = new(*mc.UseRingbuffersLocalAttn)
	}
	if mc.SingleThreadedExecution != nil {
		c.singleThreadedExecution = new(*mc.SingleThreadedExecution)
	}
	if mc.EnableDebugger != nil {
		c.enableDebugger = new(*mc.EnableDebugger)
	}
	if len(mc.SelectedSignatures) > 0 {
		c.selectedSignatures = slices.Clone(mc.SelectedSignatures)
	}
	if mc.NumThreads != nil {
		c.numThreads = new(*mc.NumThreads)
	}
	if mc.AudioNumThreads != nil {
		c.audioNumThreads = new(*mc.AudioNumThreads)
	}
	if mc.VisionDispatchLibDir != nil {
		c.visionDispatchLibDir = *mc.VisionDispatchLibDir
	}
	if mc.AudioDispatchLibDir != nil {
		c.audioDispatchLibDir = *mc.AudioDispatchLibDir
	}
	if mc.MaxInputLength != nil {
		c.maxInputLength = new(*mc.MaxInputLength)
	}
	if mc.MinInputLength != nil {
		c.minInputLength = new(*mc.MinInputLength)
	}
	if mc.LoraRank != nil {
		c.loraRank = new(*mc.LoraRank)
	}
	if len(mc.SupportedLoraRanks) > 0 {
		c.supportedLoraRanks = slices.Clone(mc.SupportedLoraRanks)
	}
	if mc.AudioLoraRank != nil {
		c.audioLoraRank = new(*mc.AudioLoraRank)
	}
	if len(mc.SupportedAudioLoraRanks) > 0 {
		c.supportedAudioLoraRanks = slices.Clone(mc.SupportedAudioLoraRanks)
	}
	if mc.LibPath != nil {
		c.libPath = *mc.LibPath
	}
	if mc.LibName != nil {
		c.libName = *mc.LibName
	}
	if mc.ModelPath != nil {
		c.modelPath = *mc.ModelPath
	}

	var sampler SamplerParams
	hasSamplerOpts := false

	if c.defaultSampler != nil {
		sampler = *c.defaultSampler
		hasSamplerOpts = true
	} else {
		sampler = DefaultSamplerParams()
	}

	if mc.SamplerType != nil {
		st, err := parseSamplerType(*mc.SamplerType)
		if err != nil {
			return err
		}
		sampler.Type = st
		hasSamplerOpts = true
	}
	if mc.TopK != nil {
		sampler.TopK = int32(*mc.TopK)
		hasSamplerOpts = true
	}
	if mc.TopP != nil {
		sampler.TopP = *mc.TopP
		hasSamplerOpts = true
	}
	if mc.Temperature != nil {
		sampler.Temperature = *mc.Temperature
		hasSamplerOpts = true
	}
	if mc.Seed != nil {
		sampler.Seed = int32(*mc.Seed)
		hasSamplerOpts = true
	}

	if hasSamplerOpts {
		c.defaultSampler = new(sampler)
	}

	return nil
}

func parseSamplerType(s string) (SamplerType, error) {
	switch strings.ToLower(s) {
	case "top_k", "topk":
		return SamplerTopK, nil
	case "top_p", "topp":
		return SamplerTopP, nil
	case "greedy":
		return SamplerGreedy, nil
	case "unspecified", "":
		return SamplerTypeUnspecified, nil
	default:
		return 0, fmt.Errorf("unknown sampler type %q (use top_k | top_p | greedy)", s)
	}
}

func parseSpeculativeDecodingMode(s string) (SpeculativeDecodingMode, error) {
	switch strings.ToLower(s) {
	case "off", "":
		return SpeculativeDecodingOff, nil
	case "on":
		return SpeculativeDecodingOn, nil
	case "adaptive":
		return SpeculativeDecodingAdaptive, nil
	default:
		return 0, fmt.Errorf("unknown speculative decoding mode %q (use off | on | adaptive)", s)
	}
}
