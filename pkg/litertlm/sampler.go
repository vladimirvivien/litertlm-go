package litertlm

import "unsafe"

// SamplerParams configures token sampling for generation. It is materialized
// into an opaque C `LiteRtLmSamplerParams*` handle when passed to
// SessionConfig.SetSamplerParams.
type SamplerParams struct {
	Type        SamplerType
	TopK        int32
	TopP        float32
	Temperature float32
	Seed        int32
}

// DefaultSamplerParams returns a reasonable greedy default. Callers may mutate
// the returned value before passing it to SessionConfig.SetSamplerParams.
func DefaultSamplerParams() SamplerParams {
	return SamplerParams{
		Type:        SamplerGreedy,
		TopK:        1,
		TopP:        1.0,
		Temperature: 0.0,
		Seed:        0,
	}
}

type samplerParamsHandle uintptr

func (h samplerParamsHandle) delete() { deleteHandle(h, samplerParamsDeleteFunc) }

func newSamplerParamsHandle(p SamplerParams) (samplerParamsHandle, error) {
	typ := int32(p.Type)
	h, err := callStatusHandle[samplerParamsHandle](
		samplerParamsCreateFunc,
		"sampler_params_create",
		unsafe.Pointer(&typ),
	)
	if err != nil {
		return 0, err
	}

	cleanupOnErr := func(err error) (samplerParamsHandle, error) {
		h.delete()
		return 0, err
	}

	// Only set numeric parameters when they apply or are explicitly populated
	// so greedy/unspecified configs preserve C-side defaults.
	if p.Type == SamplerTopK || p.Type == SamplerTopP || p.TopK != 0 {
		if err = setHandleScalar(h, "sampler params", samplerParamsSetTopKFunc, "sampler_params_set_top_k", p.TopK); err != nil {
			return cleanupOnErr(err)
		}
	}
	if p.Type == SamplerTopP || p.TopP != 0 {
		if err = setHandleScalar(h, "sampler params", samplerParamsSetTopPFunc, "sampler_params_set_top_p", p.TopP); err != nil {
			return cleanupOnErr(err)
		}
	}
	if p.Temperature != 0 {
		if err = setHandleScalar(h, "sampler params", samplerParamsSetTemperatureFunc, "sampler_params_set_temperature", p.Temperature); err != nil {
			return cleanupOnErr(err)
		}
	}
	if p.Seed != 0 {
		if err = setHandleScalar(h, "sampler params", samplerParamsSetSeedFunc, "sampler_params_set_seed", p.Seed); err != nil {
			return cleanupOnErr(err)
		}
	}
	return h, nil
}
