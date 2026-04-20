package litertlm

// SamplerParams is the Go-side image of the C `LiteRtLmSamplerParams` struct.
// Layout must match the C struct exactly:
//
//	typedef struct {
//	    Type    type;         // int32 enum
//	    int32_t top_k;
//	    float   top_p;
//	    float   temperature;
//	    int32_t seed;
//	} LiteRtLmSamplerParams;
//
// All fields are 4-byte aligned with no padding — 20 bytes total.
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
