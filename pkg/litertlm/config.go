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
	Backend                    *string  `json:"backend,omitempty"`
	MaxTokens                  *int     `json:"max_tokens,omitempty"`
	CacheDir                   *string  `json:"cache_dir,omitempty"`
	SpeculativeDecodingEnabled *bool    `json:"speculative_decoding_enabled,omitempty"`
	ParallelSectionLoading     *bool    `json:"parallel_section_loading,omitempty"`
	NumThreads                 *int     `json:"num_threads,omitempty"`
	AudioNumThreads            *int     `json:"audio_num_threads,omitempty"`
	LoraRank                   *int     `json:"lora_rank,omitempty"`
	SupportedLoraRanks         []int    `json:"supported_lora_ranks,omitempty"`
	AudioLoraRank              *int     `json:"audio_lora_rank,omitempty"`
	SupportedAudioLoraRanks    []int    `json:"supported_audio_lora_ranks,omitempty"`
	SamplerType                *string  `json:"sampler_type,omitempty"`
	Temperature                *float32 `json:"temperature,omitempty"`
	TopK                       *int     `json:"top_k,omitempty"`
	TopP                       *float32 `json:"top_p,omitempty"`
	Seed                       *int     `json:"seed,omitempty"`
	LibPath                    *string  `json:"lib_path,omitempty"`
	LibName                    *string  `json:"lib_name,omitempty"`
	ModelPath                  *string  `json:"model_path,omitempty"`
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
	if mc.ParallelSectionLoading != nil {
		c.parallelSectionLoading = new(*mc.ParallelSectionLoading)
	}
	if mc.NumThreads != nil {
		c.numThreads = new(*mc.NumThreads)
	}
	if mc.AudioNumThreads != nil {
		c.audioNumThreads = new(*mc.AudioNumThreads)
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
