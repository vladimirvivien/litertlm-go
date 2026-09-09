package litertlm

import (
	"errors"
	"fmt"
	"runtime"
	"slices"
	"unsafe"

	"github.com/vladimirvivien/litertlm-go/internal/cstr"
)

// ModelCapabilities represents metadata and runtime capabilities extracted
// from a .litertlm model file without loading full weight tensors or allocating
// KV caches.
type ModelCapabilities struct {
	Path                        string      `json:"path,omitempty"`
	SupportsSpeculativeDecoding bool        `json:"supports_speculative_decoding"`
	SupportsThinking            bool        `json:"supports_thinking"`
	SupportsFunctionCalling     bool        `json:"supports_function_calling"`
	DefaultSampler              SamplerType `json:"default_sampler"`
	DefaultTemperature          float32     `json:"default_temperature"`
	DefaultTopK                 int         `json:"default_top_k"`
	DefaultTopP                 float32     `json:"default_top_p"`
	MaxVisionTokenBudget        int         `json:"max_vision_token_budget"`
	SupportedModalities         []Modality  `json:"supported_modalities"`
}

// SupportsModality reports whether the model supports the specified input modality.
func (c *ModelCapabilities) SupportsModality(m Modality) bool {
	if c == nil {
		return false
	}
	return slices.Contains(c.SupportedModalities, m)
}

// SupportsText reports whether the model supports text input.
func (c *ModelCapabilities) SupportsText() bool {
	return c.SupportsModality(ModalityText)
}

// SupportsVision reports whether the model supports vision / image input.
func (c *ModelCapabilities) SupportsVision() bool {
	return c.SupportsModality(ModalityVision)
}

// SupportsAudio reports whether the model supports audio input.
func (c *ModelCapabilities) SupportsAudio() bool {
	return c.SupportsModality(ModalityAudio)
}

// SupportsVideo reports whether the model supports video input.
func (c *ModelCapabilities) SupportsVideo() bool {
	return c.SupportsModality(ModalityVideo)
}

// OpenLoadedFile opens a .litertlm file for capability inspection.
// The caller is responsible for calling Delete() on the returned LoadedFile handle.
func OpenLoadedFile(path string) (LoadedFile, error) {
	if path == "" {
		return 0, errors.New("litertlm: model path is required")
	}
	pathPtr, err := cstr.BytePtrFromString(path)
	if err != nil {
		return 0, err
	}
	f, err := callStatusHandle[LoadedFile](loadedFileCreateFunc, "loaded_file_create", unsafe.Pointer(&pathPtr))
	runtime.KeepAlive(pathPtr)
	return f, err
}

// InspectModel opens a .litertlm file and extracts its capabilities, supported modalities,
// and default sampling parameters.
//
// InspectModel reads only container metadata without allocating execution graphs, shaders,
// or KV caches. The underlying C handle is freed before InspectModel returns.
func InspectModel(path string) (*ModelCapabilities, error) {
	if err := Load("", "", ""); err != nil {
		return nil, fmt.Errorf("litertlm: load library for InspectModel: %w", err)
	}
	handle, err := OpenLoadedFile(path)
	if err != nil {
		return nil, fmt.Errorf("litertlm: inspect model %q: %w", path, err)
	}
	defer handle.Delete()

	return handle.Snapshot(path), nil
}

// Delete frees the C-allocated LiteRtLmLoadedFile resources.
func (f LoadedFile) Delete() { deleteHandle(f, loadedFileDeleteFunc) }

// HasSpeculativeDecodingSupport reports whether the loaded model has speculative decoding support.
func (f LoadedFile) HasSpeculativeDecodingSupport() bool {
	return getHandleBool(f, loadedFileHasSpeculativeDecodingSupportFunc, "loaded_file_has_speculative_decoding_support")
}

// SupportsThinking reports whether the model has thinking / reasoning support.
func (f LoadedFile) SupportsThinking() bool {
	return getHandleBool(f, loadedFileSupportsThinkingFunc, "loaded_file_supports_thinking")
}

// SupportsFunctionCalling reports whether the model supports function calling / tool use.
func (f LoadedFile) SupportsFunctionCalling() bool {
	return getHandleBool(f, loadedFileSupportsFunctionCallingFunc, "loaded_file_supports_function_calling")
}

// SamplerType returns the model's default sampler type.
func (f LoadedFile) SamplerType() SamplerType {
	return SamplerType(getHandleScalar[LoadedFile, int32](f, loadedFileSamplerTypeFunc, "loaded_file_sampler_type", int32(SamplerTypeUnspecified)))
}

// SamplerTemperature returns the model's default sampler temperature.
func (f LoadedFile) SamplerTemperature() float32 {
	return getHandleScalar[LoadedFile, float32](f, loadedFileSamplerTemperatureFunc, "loaded_file_sampler_temperature", 0)
}

// SamplerTopK returns the model's default sampler top_k.
func (f LoadedFile) SamplerTopK() int {
	return int(getHandleScalar[LoadedFile, int32](f, loadedFileSamplerTopKFunc, "loaded_file_sampler_top_k", 0))
}

// SamplerTopP returns the model's default sampler top_p.
func (f LoadedFile) SamplerTopP() float32 {
	return getHandleScalar[LoadedFile, float32](f, loadedFileSamplerTopPFunc, "loaded_file_sampler_top_p", 0)
}

// SupportsInputModality reports whether the model supports the specified input modality.
func (f LoadedFile) SupportsInputModality(modality Modality) bool {
	if f == 0 {
		return false
	}
	modVal := int32(modality)
	v, err := callStatusBool(loadedFileSupportsInputModalityFunc,
		"loaded_file_supports_input_modality", unsafe.Pointer(&f), unsafe.Pointer(&modVal))
	return err == nil && v
}

// MaxVisionTokenBudget returns the maximum vision token budget, or -1 if unsupported or unset.
func (f LoadedFile) MaxVisionTokenBudget() int {
	return int(getHandleScalar[LoadedFile, int32](f, loadedFileMaxVisionTokenBudgetFunc, "loaded_file_max_vision_token_budget", -1))
}

// Snapshot copies all capability properties from the handle into a Go ModelCapabilities struct.
func (f LoadedFile) Snapshot(path string) *ModelCapabilities {
	if f == 0 {
		return nil
	}
	allModalities := []Modality{ModalityText, ModalityVision, ModalityAudio, ModalityVideo}
	var supported []Modality
	for _, m := range allModalities {
		if f.SupportsInputModality(m) {
			supported = append(supported, m)
		}
	}
	return &ModelCapabilities{
		Path:                        path,
		SupportsSpeculativeDecoding: f.HasSpeculativeDecodingSupport(),
		SupportsThinking:            f.SupportsThinking(),
		SupportsFunctionCalling:     f.SupportsFunctionCalling(),
		DefaultSampler:              f.SamplerType(),
		DefaultTemperature:          f.SamplerTemperature(),
		DefaultTopK:                 f.SamplerTopK(),
		DefaultTopP:                 f.SamplerTopP(),
		MaxVisionTokenBudget:        f.MaxVisionTokenBudget(),
		SupportedModalities:         supported,
	}
}
