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
	Path                        string                     `json:"path,omitempty"`
	ModelType                   ModelType                  `json:"model_type"`
	MinRuntimeVersion           string                     `json:"min_runtime_version,omitempty"`
	SupportsSpeculativeDecoding bool                       `json:"supports_speculative_decoding"`
	SupportsThinking            bool                       `json:"supports_thinking"`
	SupportsFunctionCalling     bool                       `json:"supports_function_calling"`
	DefaultSampler              SamplerType                `json:"default_sampler"`
	DefaultTemperature          float32                    `json:"default_temperature"`
	DefaultTopK                 int                        `json:"default_top_k"`
	DefaultTopP                 float32                    `json:"default_top_p"`
	MaxContextTokens            int                        `json:"max_context_tokens,omitempty"`
	IsDynamicContext            bool                       `json:"is_dynamic_context"`
	MaxVisionTokenBudget        int                        `json:"max_vision_token_budget"`
	VisionSignatureLengths      []int                      `json:"vision_signature_lengths,omitempty"`
	EmbeddingDimension          int                        `json:"embedding_dimension,omitempty"`
	EmbeddingSignatureLengths   []int                      `json:"embedding_signature_lengths,omitempty"`
	SupportedModalities         []Modality                 `json:"supported_modalities"`
	ModalityBackends            map[Modality][]BackendType `json:"modality_backends,omitempty"`
	ModalityNPUBrand            map[Modality]NPUBrand      `json:"modality_npu_brand,omitempty"`
	ModalitySoCName             map[Modality]string        `json:"modality_soc_name,omitempty"`
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

// ModelType returns the loaded model kind (LLM, Embedding, or Unknown).
func (f LoadedFile) ModelType() (ModelType, error) {
	v, err := getHandleScalarErr[LoadedFile, int32](f, "loaded file", loadedFileModelTypeFunc, "loaded_file_model_type")
	if err != nil {
		return ModelTypeUnknown, err
	}
	return ModelType(v), nil
}

// MaxContextTokens returns the model's context capacity in tokens, or
// ErrNotFound when the metadata does not specify one.
func (f LoadedFile) MaxContextTokens() (int, error) {
	v, err := getHandleScalarErr[LoadedFile, uint32](f, "loaded file", loadedFileMaxContextTokensFunc, "loaded_file_max_context_tokens")
	if err != nil {
		return 0, err
	}
	return int(v), nil
}

// IsDynamicContext reports whether the model graph supports configurable context lengths.
func (f LoadedFile) IsDynamicContext() bool {
	return getHandleBool(f, loadedFileIsDynamicContextFunc, "loaded_file_is_dynamic_context")
}

// VisionSignatureLengths returns supported discrete vision token budgets.
// Returns ErrNotFound if the model does not support vision or defines none.
func (f LoadedFile) VisionSignatureLengths() ([]int, error) {
	return f.queryInt32Slice(loadedFileVisionSignatureSelectionFunc, "loaded_file_vision_signature_selection")
}

// EmbeddingDimension returns the output embedding dimension, or ErrNotFound
// for non-embedding models.
func (f LoadedFile) EmbeddingDimension() (int, error) {
	v, err := getHandleScalarErr[LoadedFile, int32](f, "loaded file", loadedFileEmbeddingDimensionFunc, "loaded_file_embedding_dimension")
	if err != nil {
		return 0, err
	}
	return int(v), nil
}

// EmbeddingSignatureLengths returns supported embedding sequence lengths, or
// ErrNotFound for non-embedding models.
func (f LoadedFile) EmbeddingSignatureLengths() ([]int, error) {
	return f.queryInt32Slice(loadedFileEmbeddingSignatureSelectionFunc, "loaded_file_embedding_signature_selection")
}

func (f LoadedFile) queryInt32Slice(fn *lazyFun, op string) ([]int, error) {
	if f == 0 {
		return nil, fmt.Errorf("litertlm: %s: invalid loaded file", op)
	}
	raw, err := queryCBuffer(func(buf *int32, maxSize uint64) (uint64, error) {
		return callStatusScalar[uint64](fn, op,
			unsafe.Pointer(&f), unsafe.Pointer(&buf), unsafe.Pointer(&maxSize))
	})
	if err != nil {
		return nil, err
	}
	if len(raw) == 0 {
		return nil, nil
	}
	out := make([]int, len(raw))
	for i, v := range raw {
		out[i] = int(v)
	}
	return out, nil
}

// ModalitySupportedBackends returns the priority-ordered hardware backends
// supported for modality (nil if modality is unsupported).
func (f LoadedFile) ModalitySupportedBackends(modality Modality) ([]BackendType, error) {
	if f == 0 {
		return nil, fmt.Errorf("litertlm: loaded_file_modality_supported_backends: invalid loaded file")
	}
	modVal := int32(modality)
	return queryCBuffer(func(buf *BackendType, maxSize uint64) (uint64, error) {
		return callStatusScalar[uint64](loadedFileModalitySupportedBackendsFunc, "loaded_file_modality_supported_backends",
			unsafe.Pointer(&f), unsafe.Pointer(&modVal), unsafe.Pointer(&buf), unsafe.Pointer(&maxSize))
	})
}

// ModalityNPUBrand returns the detected NPU vendor for modality, or
// NPUBrandUnknown if not NPU-compiled.
func (f LoadedFile) ModalityNPUBrand(modality Modality) (NPUBrand, error) {
	if f == 0 {
		return NPUBrandUnknown, fmt.Errorf("litertlm: loaded_file_modality_npu_brand: invalid loaded file")
	}
	modVal := int32(modality)
	v, err := callStatusScalar[int32](loadedFileModalityNpuBrandFunc, "loaded_file_modality_npu_brand",
		unsafe.Pointer(&f), unsafe.Pointer(&modVal))
	if err != nil {
		return NPUBrandUnknown, err
	}
	return NPUBrand(v), nil
}

// ModalitySoCName returns the target SoC identifier for modality (empty string
// when unspecified).
func (f LoadedFile) ModalitySoCName(modality Modality) (string, error) {
	if f == 0 {
		return "", fmt.Errorf("litertlm: loaded_file_modality_soc_name: invalid loaded file")
	}
	modVal := int32(modality)
	s, _, err := callStatusString(loadedFileModalitySocNameFunc, "loaded_file_modality_soc_name",
		unsafe.Pointer(&f), unsafe.Pointer(&modVal))
	if err != nil {
		return "", err
	}
	return s, nil
}

// MinRuntimeVersion returns the minimum LiteRT-LM runtime version string
// required by the model (empty string when unspecified).
func (f LoadedFile) MinRuntimeVersion() (string, error) {
	return getHandleStringErr(f, "loaded file", loadedFileMinRuntimeVersionFunc, "loaded_file_min_runtime_version")
}

// Snapshot copies all capability properties from the handle into a Go ModelCapabilities struct.
func (f LoadedFile) Snapshot(path string) *ModelCapabilities {
	if f == 0 {
		return nil
	}
	allModalities := []Modality{ModalityText, ModalityVision, ModalityAudio, ModalityVideo}
	var (
		supported        []Modality
		modalityBackends map[Modality][]BackendType
		modalityNPUBrand map[Modality]NPUBrand
		modalitySoCName  map[Modality]string
	)
	for _, m := range allModalities {
		if !f.SupportsInputModality(m) {
			continue
		}
		supported = append(supported, m)
		if backends, err := f.ModalitySupportedBackends(m); err == nil && len(backends) > 0 {
			if modalityBackends == nil {
				modalityBackends = make(map[Modality][]BackendType)
			}
			modalityBackends[m] = backends
		}
		if brand, err := f.ModalityNPUBrand(m); err == nil && brand != NPUBrandUnknown {
			if modalityNPUBrand == nil {
				modalityNPUBrand = make(map[Modality]NPUBrand)
			}
			modalityNPUBrand[m] = brand
		}
		if soc, err := f.ModalitySoCName(m); err == nil && soc != "" {
			if modalitySoCName == nil {
				modalitySoCName = make(map[Modality]string)
			}
			modalitySoCName[m] = soc
		}
	}

	modelType, _ := f.ModelType()
	minVer, _ := f.MinRuntimeVersion()
	maxCtx, _ := f.MaxContextTokens()
	visionLens, _ := f.VisionSignatureLengths()
	embedDim, _ := f.EmbeddingDimension()
	embedLens, _ := f.EmbeddingSignatureLengths()

	return &ModelCapabilities{
		Path:                        path,
		ModelType:                   modelType,
		MinRuntimeVersion:           minVer,
		SupportsSpeculativeDecoding: f.HasSpeculativeDecodingSupport(),
		SupportsThinking:            f.SupportsThinking(),
		SupportsFunctionCalling:     f.SupportsFunctionCalling(),
		DefaultSampler:              f.SamplerType(),
		DefaultTemperature:          f.SamplerTemperature(),
		DefaultTopK:                 f.SamplerTopK(),
		DefaultTopP:                 f.SamplerTopP(),
		MaxContextTokens:            maxCtx,
		IsDynamicContext:            f.IsDynamicContext(),
		MaxVisionTokenBudget:        f.MaxVisionTokenBudget(),
		VisionSignatureLengths:      visionLens,
		EmbeddingDimension:          embedDim,
		EmbeddingSignatureLengths:   embedLens,
		SupportedModalities:         supported,
		ModalityBackends:            modalityBackends,
		ModalityNPUBrand:            modalityNPUBrand,
		ModalitySoCName:             modalitySoCName,
	}
}
