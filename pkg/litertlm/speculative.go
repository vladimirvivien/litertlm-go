package litertlm

import "math"

// SpeculativeDecodingConfig configures multi-token speculative decoding on an
// engine or session. Zero Lenience uses the lossless runtime default; values in
// (0, 1] relax draft acceptance in exchange for speed and require a runtime
// exporting litert_lm_speculative_decoding_config_set_lenience (v0.20.0+).
type SpeculativeDecodingConfig struct {
	Mode     SpeculativeDecodingMode
	Lenience float32
}

type speculativeDecodingConfigHandle uintptr

func (h speculativeDecodingConfigHandle) delete() {
	deleteHandle(h, speculativeDecodingConfigDeleteFunc)
}

func needsLenienceCall(lenience float32) bool {
	return lenience != 0 || math.IsNaN(float64(lenience))
}

func newSpeculativeDecodingConfigHandle(cfg SpeculativeDecodingConfig) (speculativeDecodingConfigHandle, error) {
	h, err := callStatusHandle[speculativeDecodingConfigHandle](
		speculativeDecodingConfigCreateFunc,
		"speculative_decoding_config_create",
	)
	if err != nil {
		return 0, err
	}
	if err = setHandleScalar(h, "speculative decoding config", speculativeDecodingConfigSetModeFunc,
		"speculative_decoding_config_set_mode", int32(cfg.Mode)); err != nil {
		h.delete()
		return 0, err
	}
	if needsLenienceCall(cfg.Lenience) {
		if err = setHandleScalar(h, "speculative decoding config", speculativeDecodingConfigSetLenienceFunc,
			"speculative_decoding_config_set_lenience", cfg.Lenience); err != nil {
			h.delete()
			return 0, err
		}
	}
	return h, nil
}

// setSpeculativeDecodingConfig attaches cfg to any engine or session config handle.
func setSpeculativeDecodingConfig[H cHandle](
	h H,
	handleName string,
	f *lazyFun,
	op string,
	cfg SpeculativeDecodingConfig,
) error {
	if h == 0 {
		return setHandleConfig[H, speculativeDecodingConfigHandle](0, handleName, f, op, 0)
	}
	return withTempHandle(
		func() (speculativeDecodingConfigHandle, error) {
			return newSpeculativeDecodingConfigHandle(cfg)
		},
		func(ch speculativeDecodingConfigHandle) { ch.delete() },
		func(ch speculativeDecodingConfigHandle) error {
			return setHandleConfig(h, handleName, f, op, ch)
		},
	)
}
