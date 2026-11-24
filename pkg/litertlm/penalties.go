package litertlm

// RepetitionPenaltyConfig configures per-turn repetition, presence, and
// frequency penalties applied to output logits during decoding.
type RepetitionPenaltyConfig struct {
	RepetitionPenalty float32
	PresencePenalty   float32
	FrequencyPenalty  float32
	WindowSize        int
}

// NoRepeatNgramConfig bans repeating token n-grams of NgramSize within WindowSize
// history tokens.
type NoRepeatNgramConfig struct {
	NgramSize  int
	WindowSize int
}

// SuppressTokensConfig guarantees the listed token IDs receive -inf logit scores
// so they are never sampled during generation.
type SuppressTokensConfig struct {
	TokenIDs []int
}

type (
	repetitionPenaltyConfigHandle uintptr
	noRepeatNgramConfigHandle     uintptr
	suppressTokensConfigHandle    uintptr
)

func (h repetitionPenaltyConfigHandle) delete() { deleteHandle(h, repetitionPenaltyConfigDeleteFunc) }
func (h noRepeatNgramConfigHandle) delete()     { deleteHandle(h, noRepeatNgramConfigDeleteFunc) }
func (h suppressTokensConfigHandle) delete()    { deleteHandle(h, suppressTokensConfigDeleteFunc) }

func newRepetitionPenaltyHandle(cfg RepetitionPenaltyConfig) (repetitionPenaltyConfigHandle, error) {
	h, err := callStatusHandle[repetitionPenaltyConfigHandle](
		repetitionPenaltyConfigCreateFunc,
		"repetition_penalty_config_create",
	)
	if err != nil {
		return 0, err
	}
	if cfg.RepetitionPenalty != 0 {
		if err = setHandleScalar(h, "repetition penalty config",
			repetitionPenaltyConfigSetRepetitionPenaltyFunc,
			"repetition_penalty_config_set_repetition_penalty", cfg.RepetitionPenalty); err != nil {
			h.delete()
			return 0, err
		}
	}
	if cfg.PresencePenalty != 0 {
		if err = setHandleScalar(h, "repetition penalty config",
			repetitionPenaltyConfigSetPresencePenaltyFunc,
			"repetition_penalty_config_set_presence_penalty", cfg.PresencePenalty); err != nil {
			h.delete()
			return 0, err
		}
	}
	if cfg.FrequencyPenalty != 0 {
		if err = setHandleScalar(h, "repetition penalty config",
			repetitionPenaltyConfigSetFrequencyPenaltyFunc,
			"repetition_penalty_config_set_frequency_penalty", cfg.FrequencyPenalty); err != nil {
			h.delete()
			return 0, err
		}
	}
	if cfg.WindowSize != 0 {
		if err = setHandleScalar(h, "repetition penalty config",
			repetitionPenaltyConfigSetWindowSizeFunc,
			"repetition_penalty_config_set_window_size", int32(cfg.WindowSize)); err != nil {
			h.delete()
			return 0, err
		}
	}
	return h, nil
}

func newNoRepeatNgramHandle(cfg NoRepeatNgramConfig) (noRepeatNgramConfigHandle, error) {
	h, err := callStatusHandle[noRepeatNgramConfigHandle](
		noRepeatNgramConfigCreateFunc,
		"no_repeat_ngram_config_create",
	)
	if err != nil {
		return 0, err
	}
	if cfg.NgramSize != 0 {
		if err = setHandleScalar(h, "no repeat ngram config",
			noRepeatNgramConfigSetNoRepeatNgramSizeFunc,
			"no_repeat_ngram_config_set_no_repeat_ngram_size", int32(cfg.NgramSize)); err != nil {
			h.delete()
			return 0, err
		}
	}
	if cfg.WindowSize != 0 {
		if err = setHandleScalar(h, "no repeat ngram config",
			noRepeatNgramConfigSetWindowSizeFunc,
			"no_repeat_ngram_config_set_window_size", int32(cfg.WindowSize)); err != nil {
			h.delete()
			return 0, err
		}
	}
	return h, nil
}

func newSuppressTokensHandle(cfg SuppressTokensConfig) (suppressTokensConfigHandle, error) {
	h, err := callStatusHandle[suppressTokensConfigHandle](
		suppressTokensConfigCreateFunc,
		"suppress_tokens_config_create",
	)
	if err != nil {
		return 0, err
	}
	if err = setHandleInt32Slice(h, "suppress tokens config",
		suppressTokensConfigSetSuppressTokensFunc,
		"suppress_tokens_config_set_suppress_tokens", cfg.TokenIDs); err != nil {
		h.delete()
		return 0, err
	}
	return h, nil
}
