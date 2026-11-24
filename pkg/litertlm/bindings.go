package litertlm

import "github.com/jupiterrider/ffi"

// ffiTypeSizeT aliases the ffi type used for C `size_t`. On all supported
// platforms this is a 64-bit unsigned integer.
var ffiTypeSizeT = ffi.TypeUint64

// Every exported litert_lm_* function in the LiteRT-LM C API (1.0.0+) has a
// package-level *lazyFun declared here. Each one resolves its C symbol against
// libHandle on first call, returning StatusUnimplemented if the loaded
// library does not export it.

var (
	// ---- API Version & Error Reporter ----
	getLastErrorMessageFunc = newLazyFun(
		"litert_lm_get_last_error_message",
		&ffi.TypePointer)
	clearLastErrorFunc = newLazyFun(
		"litert_lm_clear_last_error",
		&ffi.TypeVoid)

	// ---- Sampler Params ----
	samplerParamsCreateFunc = newLazyFun(
		"litert_lm_sampler_params_create",
		&ffi.TypeSint32,
		&ffi.TypeSint32,  // type
		&ffi.TypePointer, // out_params
	)
	samplerParamsDeleteFunc = newLazyFun(
		"litert_lm_sampler_params_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	samplerParamsSetTopKFunc = newLazyFun(
		"litert_lm_sampler_params_set_top_k",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	samplerParamsSetTopPFunc = newLazyFun(
		"litert_lm_sampler_params_set_top_p",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeFloat)
	samplerParamsSetTemperatureFunc = newLazyFun(
		"litert_lm_sampler_params_set_temperature",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeFloat)
	samplerParamsSetSeedFunc = newLazyFun(
		"litert_lm_sampler_params_set_seed",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)

	// ---- Speculative Decoding Config ----
	speculativeDecodingConfigCreateFunc = newLazyFun(
		"litert_lm_speculative_decoding_config_create",
		&ffi.TypeSint32, &ffi.TypePointer)
	speculativeDecodingConfigDeleteFunc = newLazyFun(
		"litert_lm_speculative_decoding_config_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	speculativeDecodingConfigSetModeFunc = newLazyFun(
		"litert_lm_speculative_decoding_config_set_mode",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	speculativeDecodingConfigSetLenienceFunc = newLazyFun(
		"litert_lm_speculative_decoding_config_set_lenience",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeFloat)

	// ---- Logit Penalty & Thinking Configs ----
	repetitionPenaltyConfigCreateFunc = newLazyFun(
		"litert_lm_repetition_penalty_config_create",
		&ffi.TypeSint32, &ffi.TypePointer)
	repetitionPenaltyConfigDeleteFunc = newLazyFun(
		"litert_lm_repetition_penalty_config_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	repetitionPenaltyConfigSetRepetitionPenaltyFunc = newLazyFun(
		"litert_lm_repetition_penalty_config_set_repetition_penalty",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeFloat)
	repetitionPenaltyConfigSetPresencePenaltyFunc = newLazyFun(
		"litert_lm_repetition_penalty_config_set_presence_penalty",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeFloat)
	repetitionPenaltyConfigSetFrequencyPenaltyFunc = newLazyFun(
		"litert_lm_repetition_penalty_config_set_frequency_penalty",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeFloat)
	repetitionPenaltyConfigSetWindowSizeFunc = newLazyFun(
		"litert_lm_repetition_penalty_config_set_window_size",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)

	noRepeatNgramConfigCreateFunc = newLazyFun(
		"litert_lm_no_repeat_ngram_config_create",
		&ffi.TypeSint32, &ffi.TypePointer)
	noRepeatNgramConfigDeleteFunc = newLazyFun(
		"litert_lm_no_repeat_ngram_config_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	noRepeatNgramConfigSetNoRepeatNgramSizeFunc = newLazyFun(
		"litert_lm_no_repeat_ngram_config_set_no_repeat_ngram_size",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	noRepeatNgramConfigSetWindowSizeFunc = newLazyFun(
		"litert_lm_no_repeat_ngram_config_set_window_size",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)

	suppressTokensConfigCreateFunc = newLazyFun(
		"litert_lm_suppress_tokens_config_create",
		&ffi.TypeSint32, &ffi.TypePointer)
	suppressTokensConfigDeleteFunc = newLazyFun(
		"litert_lm_suppress_tokens_config_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	suppressTokensConfigSetSuppressTokensFunc = newLazyFun(
		"litert_lm_suppress_tokens_config_set_suppress_tokens",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffiTypeSizeT)

	thinkingConfigCreateFunc = newLazyFun(
		"litert_lm_thinking_config_create",
		&ffi.TypeSint32, &ffi.TypePointer)
	thinkingConfigDeleteFunc = newLazyFun(
		"litert_lm_thinking_config_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	thinkingConfigSetEnableThinkingFunc = newLazyFun(
		"litert_lm_thinking_config_set_enable_thinking",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeUint8)
	thinkingConfigSetThinkingTokenBudgetFunc = newLazyFun(
		"litert_lm_thinking_config_set_thinking_token_budget",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)

	// ---- Session config ----
	sessionConfigCreateFunc = newLazyFun(
		"litert_lm_session_config_create",
		&ffi.TypeSint32, &ffi.TypePointer)
	sessionConfigSetMaxOutputTokensFunc = newLazyFun(
		"litert_lm_session_config_set_max_output_tokens",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	sessionConfigSetSamplerParamsFunc = newLazyFun(
		"litert_lm_session_config_set_sampler_params",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	sessionConfigSetApplyPromptTemplateFunc = newLazyFun(
		"litert_lm_session_config_set_apply_prompt_template",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeUint8)
	sessionConfigSetLoraPathFunc = newLazyFun(
		"litert_lm_session_config_set_lora_path",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	sessionConfigSetAudioLoraPathFunc = newLazyFun(
		"litert_lm_session_config_set_audio_lora_path",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	sessionConfigSetSpeculativeDecodingConfigFunc = newLazyFun(
		"litert_lm_session_config_set_speculative_decoding_config",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	sessionConfigDeleteFunc = newLazyFun(
		"litert_lm_session_config_delete",
		&ffi.TypeVoid, &ffi.TypePointer)

	// ---- Conversation config ----
	conversationConfigCreateFunc = newLazyFun(
		"litert_lm_conversation_config_create",
		&ffi.TypeSint32, &ffi.TypePointer)
	conversationConfigSetSessionConfigFunc = newLazyFun(
		"litert_lm_conversation_config_set_session_config",
		&ffi.TypeSint32,
		&ffi.TypePointer, // config
		&ffi.TypePointer, // session_config
	)
	conversationConfigSetSystemMessageFunc = newLazyFun(
		"litert_lm_conversation_config_set_system_message",
		&ffi.TypeSint32,
		&ffi.TypePointer, // config
		&ffi.TypePointer, // system_message_json (const char*)
	)
	conversationConfigSetToolsFunc = newLazyFun(
		"litert_lm_conversation_config_set_tools",
		&ffi.TypeSint32,
		&ffi.TypePointer, // config
		&ffi.TypePointer, // tools_json (const char*)
	)
	conversationConfigSetMessagesFunc = newLazyFun(
		"litert_lm_conversation_config_set_messages",
		&ffi.TypeSint32,
		&ffi.TypePointer, // config
		&ffi.TypePointer, // messages_json (const char*)
	)
	conversationConfigSetEnableConstrainedDecodingFunc = newLazyFun(
		"litert_lm_conversation_config_set_enable_constrained_decoding",
		&ffi.TypeSint32,
		&ffi.TypePointer, // config
		&ffi.TypeUint8,   // enable_constrained_decoding (bool)
	)
	conversationConfigSetExtraContextFunc = newLazyFun(
		"litert_lm_conversation_config_set_extra_context",
		&ffi.TypeSint32,
		&ffi.TypePointer, // config
		&ffi.TypePointer, // extra_context_json (const char*)
	)
	conversationConfigSetFilterChannelContentFromKVCacheFunc = newLazyFun(
		"litert_lm_conversation_config_set_filter_channel_content_from_kv_cache",
		&ffi.TypeSint32,
		&ffi.TypePointer, // config
		&ffi.TypeUint8,   // filter_channel_content_from_kv_cache (bool)
	)
	conversationConfigSetStreamToolCallsFunc = newLazyFun(
		"litert_lm_conversation_config_set_stream_tool_calls",
		&ffi.TypeSint32,
		&ffi.TypePointer, // config
		&ffi.TypeUint8,   // stream_tool_calls (bool)
		&ffi.TypePointer, // channel_name (const char*)
	)
	conversationConfigSetPromptTemplateFunc = newLazyFun(
		"litert_lm_conversation_config_set_prompt_template",
		&ffi.TypeSint32,
		&ffi.TypePointer, // config
		&ffi.TypePointer, // prompt_template (const char*)
	)
	conversationConfigSetConstraintProviderFunc = newLazyFun(
		"litert_lm_conversation_config_set_constraint_provider",
		&ffi.TypeSint32,
		&ffi.TypePointer, // config
		&ffi.TypeSint32,  // provider_type (LiteRtLmConstraintProviderType)
	)
	conversationConfigSetThinkingConfigFunc = newLazyFun(
		"litert_lm_conversation_config_set_thinking_config",
		&ffi.TypeSint32,
		&ffi.TypePointer, // config
		&ffi.TypePointer, // thinking_config (const LiteRtLmThinkingConfig*)
	)
	conversationConfigDeleteFunc = newLazyFun(
		"litert_lm_conversation_config_delete",
		&ffi.TypeVoid, &ffi.TypePointer)

	// ---- ConversationOptionalArgs ----
	conversationOptionalArgsCreateFunc = newLazyFun(
		"litert_lm_conversation_optional_args_create",
		&ffi.TypeSint32, &ffi.TypePointer)
	conversationOptionalArgsDeleteFunc = newLazyFun(
		"litert_lm_conversation_optional_args_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	conversationOptionalArgsSetVisualTokenBudgetFunc = newLazyFun(
		"litert_lm_conversation_optional_args_set_visual_token_budget",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	conversationOptionalArgsSetMaxOutputTokensFunc = newLazyFun(
		"litert_lm_conversation_optional_args_set_max_output_tokens",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	conversationOptionalArgsSetRepetitionPenaltyConfigFunc = newLazyFun(
		"litert_lm_conversation_optional_args_set_repetition_penalty_config",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	conversationOptionalArgsSetNoRepeatNgramConfigFunc = newLazyFun(
		"litert_lm_conversation_optional_args_set_no_repeat_ngram_config",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	conversationOptionalArgsSetSuppressTokensConfigFunc = newLazyFun(
		"litert_lm_conversation_optional_args_set_suppress_tokens_config",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	conversationOptionalArgsSetThinkingConfigFunc = newLazyFun(
		"litert_lm_conversation_optional_args_set_thinking_config",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	conversationOptionalArgsSetConstraintFunc = newLazyFun(
		"litert_lm_conversation_optional_args_set_constraint",
		&ffi.TypeSint32,
		&ffi.TypePointer, // optional_args
		&ffi.TypeSint32,  // constraint_type (LiteRtLmConstraintType)
		&ffi.TypePointer, // constraint_string (const char*)
	)
	conversationOptionalArgsSetToolChoiceFunc = newLazyFun(
		"litert_lm_conversation_optional_args_set_tool_choice",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)

	// ---- Logging ----
	setMinLogLevelFunc = newLazyFun(
		"litert_lm_set_min_log_level",
		&ffi.TypeSint32, &ffi.TypeSint32)

	// ---- Engine settings ----
	engineSettingsCreateFunc = newLazyFun(
		"litert_lm_engine_settings_create",
		&ffi.TypeSint32,
		&ffi.TypePointer, // model_path
		&ffi.TypePointer, // backend
		&ffi.TypePointer, // vision_backend (nullable)
		&ffi.TypePointer, // audio_backend (nullable)
		&ffi.TypePointer, // out_settings
	)
	engineSettingsDeleteFunc = newLazyFun(
		"litert_lm_engine_settings_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	engineSettingsSetMaxNumTokensFunc = newLazyFun(
		"litert_lm_engine_settings_set_max_num_tokens",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	engineSettingsSetParallelFileSectionLoadingFunc = newLazyFun(
		"litert_lm_engine_settings_set_parallel_file_section_loading",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeUint8)
	engineSettingsSetCacheDirFunc = newLazyFun(
		"litert_lm_engine_settings_set_cache_dir",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	engineSettingsSetActivationDataTypeFunc = newLazyFun(
		"litert_lm_engine_settings_set_activation_data_type",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	engineSettingsSetPrefillChunkSizeFunc = newLazyFun(
		"litert_lm_engine_settings_set_prefill_chunk_size",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	engineSettingsEnableBenchmarkFunc = newLazyFun(
		"litert_lm_engine_settings_enable_benchmark",
		&ffi.TypeSint32, &ffi.TypePointer)
	engineSettingsSetNumPrefillTokensFunc = newLazyFun(
		"litert_lm_engine_settings_set_num_prefill_tokens",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	engineSettingsSetNumDecodeTokensFunc = newLazyFun(
		"litert_lm_engine_settings_set_num_decode_tokens",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	engineSettingsSetEnableSpeculativeDecodingFunc = newLazyFun(
		"litert_lm_engine_settings_set_enable_speculative_decoding",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeUint8)
	engineSettingsSetMaxNumImagesFunc = newLazyFun(
		"litert_lm_engine_settings_set_max_num_images",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	engineSettingsSetLitertDispatchLibDirFunc = newLazyFun(
		"litert_lm_engine_settings_set_litert_dispatch_lib_dir",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	engineSettingsCreateFromRawFileDescriptorFunc = newLazyFun(
		"litert_lm_engine_settings_create_from_raw_file_descriptor",
		&ffi.TypeSint32,
		&ffi.TypeSint32,  // fd
		&ffi.TypePointer, // backend
		&ffi.TypePointer, // vision_backend
		&ffi.TypePointer, // audio_backend
		&ffi.TypePointer, // out_settings
	)
	engineSettingsSetNumThreadsFunc = newLazyFun(
		"litert_lm_engine_settings_set_num_threads",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	engineSettingsSetAudioNumThreadsFunc = newLazyFun(
		"litert_lm_engine_settings_set_audio_num_threads",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	engineSettingsSetLoraRankFunc = newLazyFun(
		"litert_lm_engine_settings_set_lora_rank",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	engineSettingsSetSupportedLoraRanksFunc = newLazyFun(
		"litert_lm_engine_settings_set_supported_lora_ranks",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffiTypeSizeT)
	engineSettingsSetAudioLoraRankFunc = newLazyFun(
		"litert_lm_engine_settings_set_audio_lora_rank",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	engineSettingsSetSupportedAudioLoraRanksFunc = newLazyFun(
		"litert_lm_engine_settings_set_supported_audio_lora_ranks",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffiTypeSizeT)
	engineSettingsSetEnableYnnpackFunc = newLazyFun(
		"litert_lm_engine_settings_set_enable_ynnpack",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeUint8)
	engineSettingsSetMaxVisionTokensPerImageFunc = newLazyFun(
		"litert_lm_engine_settings_set_max_vision_tokens_per_image",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	engineSettingsSetSpeculativeDecodingConfigFunc = newLazyFun(
		"litert_lm_engine_settings_set_speculative_decoding_config",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	engineSettingsSetGpuEnableMetalResidencySetFunc = newLazyFun(
		"litert_lm_engine_settings_set_gpu_enable_metal_residency_set",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeUint8)
	engineSettingsSetGpuDecodeStepsPerSyncFunc = newLazyFun(
		"litert_lm_engine_settings_set_gpu_decode_steps_per_sync",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	engineSettingsSetGpuWaitForWeightUploadsFunc = newLazyFun(
		"litert_lm_engine_settings_set_gpu_wait_for_weight_uploads",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeUint8)
	engineSettingsSetUseRingbuffersLocalAttentionFunc = newLazyFun(
		"litert_lm_engine_settings_set_use_ringbuffers_local_attention",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeUint8)
	engineSettingsSetSingleThreadedExecutionFunc = newLazyFun(
		"litert_lm_engine_settings_set_single_threaded_execution",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeUint8)
	engineSettingsSetSelectedSignaturesFunc = newLazyFun(
		"litert_lm_engine_settings_set_selected_signatures",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffiTypeSizeT)

	// ---- Engine ----
	engineCreateFunc = newLazyFun(
		"litert_lm_engine_create",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	engineDeleteFunc = newLazyFun(
		"litert_lm_engine_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	engineCreateSessionFunc = newLazyFun(
		"litert_lm_engine_create_session",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer)

	// ---- Input Data ----
	inputDataCreateFunc = newLazyFun(
		"litert_lm_input_data_create",
		&ffi.TypeSint32,
		&ffi.TypeSint32,  // type
		&ffi.TypePointer, // data
		&ffiTypeSizeT,    // size
		&ffi.TypePointer, // out_input_data
	)
	inputDataDeleteFunc = newLazyFun(
		"litert_lm_input_data_delete",
		&ffi.TypeVoid,
		&ffi.TypePointer,
	)

	// ---- Session ----
	sessionDeleteFunc = newLazyFun(
		"litert_lm_session_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	sessionGenerateContentFunc = newLazyFun(
		"litert_lm_session_generate_content",
		&ffi.TypeSint32,
		&ffi.TypePointer, // session
		&ffi.TypePointer, // inputs
		&ffiTypeSizeT,    // num_inputs
		&ffi.TypePointer, // out_responses
	)
	sessionGenerateContentStreamFunc = newLazyFun(
		"litert_lm_session_generate_content_stream",
		&ffi.TypeSint32,
		&ffi.TypePointer, // session
		&ffi.TypePointer, // inputs
		&ffiTypeSizeT,    // num_inputs
		&ffi.TypePointer, // callback
		&ffi.TypePointer, // callback_data
	)
	sessionGetBenchmarkInfoFunc = newLazyFun(
		"litert_lm_session_get_benchmark_info",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	sessionCancelProcessFunc = newLazyFun(
		"litert_lm_session_cancel_process",
		&ffi.TypeSint32, &ffi.TypePointer)
	sessionRunPrefillFunc = newLazyFun(
		"litert_lm_session_run_prefill",
		&ffi.TypeSint32,
		&ffi.TypePointer, // session
		&ffi.TypePointer, // inputs
		&ffiTypeSizeT,    // num_inputs
	)
	sessionRunDecodeFunc = newLazyFun(
		"litert_lm_session_run_decode",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	sessionRunDecodeAsyncFunc = newLazyFun(
		"litert_lm_session_run_decode_async",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer)
	sessionCloneFunc = newLazyFun(
		"litert_lm_session_clone",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	sessionSaveCheckpointFunc = newLazyFun(
		"litert_lm_session_save_checkpoint",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	sessionRewindToCheckpointFunc = newLazyFun(
		"litert_lm_session_rewind_to_checkpoint",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	sessionRewindToStepFunc = newLazyFun(
		"litert_lm_session_rewind_to_step",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	sessionRunTextScoringFunc = newLazyFun(
		"litert_lm_session_run_text_scoring",
		&ffi.TypeSint32,
		&ffi.TypePointer, // session
		&ffi.TypePointer, // target_text (const char**)
		&ffiTypeSizeT,    // num_targets
		&ffi.TypeUint8,   // store_token_lengths
		&ffi.TypePointer, // out_responses
	)

	// ---- Stream Chunk ----
	streamChunkGetTextFunc = newLazyFun(
		"litert_lm_stream_chunk_get_text",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	streamChunkIsFinalFunc = newLazyFun(
		"litert_lm_stream_chunk_is_final",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	streamChunkGetErrorFunc = newLazyFun(
		"litert_lm_stream_chunk_get_error",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)

	// ---- Responses ----
	responsesDeleteFunc = newLazyFun(
		"litert_lm_responses_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	responsesGetNumCandidatesFunc = newLazyFun(
		"litert_lm_responses_get_num_candidates",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	responsesGetResponseTextAtFunc = newLazyFun(
		"litert_lm_responses_get_response_text_at",
		&ffi.TypeSint32, &ffi.TypePointer, &ffiTypeSizeT, &ffi.TypePointer)
	responsesHasScoreAtFunc = newLazyFun(
		"litert_lm_responses_has_score_at",
		&ffi.TypeSint32, &ffi.TypePointer, &ffiTypeSizeT, &ffi.TypePointer)
	responsesGetScoreAtFunc = newLazyFun(
		"litert_lm_responses_get_score_at",
		&ffi.TypeSint32, &ffi.TypePointer, &ffiTypeSizeT, &ffi.TypePointer)
	responsesHasTokenLengthAtFunc = newLazyFun(
		"litert_lm_responses_has_token_length_at",
		&ffi.TypeSint32, &ffi.TypePointer, &ffiTypeSizeT, &ffi.TypePointer)
	responsesHasTokenScoresAtFunc = newLazyFun(
		"litert_lm_responses_has_token_scores_at",
		&ffi.TypeSint32, &ffi.TypePointer, &ffiTypeSizeT, &ffi.TypePointer)
	responsesGetNumTokenScoresAtFunc = newLazyFun(
		"litert_lm_responses_get_num_token_scores_at",
		&ffi.TypeSint32, &ffi.TypePointer, &ffiTypeSizeT, &ffi.TypePointer)
	responsesGetTokenScoresAtFunc = newLazyFun(
		"litert_lm_responses_get_token_scores_at",
		&ffi.TypeSint32, &ffi.TypePointer, &ffiTypeSizeT, &ffi.TypePointer)
	responsesGetTokenLengthAtFunc = newLazyFun(
		"litert_lm_responses_get_token_length_at",
		&ffi.TypeSint32, &ffi.TypePointer, &ffiTypeSizeT, &ffi.TypePointer)

	// ---- Benchmark info ----
	benchmarkInfoDeleteFunc = newLazyFun(
		"litert_lm_benchmark_info_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	benchmarkInfoGetTimeToFirstTokenFunc = newLazyFun(
		"litert_lm_benchmark_info_get_time_to_first_token",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	benchmarkInfoGetTotalInitTimeInSecondFunc = newLazyFun(
		"litert_lm_benchmark_info_get_total_init_time_in_second",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	benchmarkInfoGetNumPrefillTurnsFunc = newLazyFun(
		"litert_lm_benchmark_info_get_num_prefill_turns",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	benchmarkInfoGetNumDecodeTurnsFunc = newLazyFun(
		"litert_lm_benchmark_info_get_num_decode_turns",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	benchmarkInfoGetPrefillTokenCountAtFunc = newLazyFun(
		"litert_lm_benchmark_info_get_prefill_token_count_at",
		&ffi.TypeSint32, &ffi.TypePointer, &ffiTypeSizeT, &ffi.TypePointer)
	benchmarkInfoGetDecodeTokenCountAtFunc = newLazyFun(
		"litert_lm_benchmark_info_get_decode_token_count_at",
		&ffi.TypeSint32, &ffi.TypePointer, &ffiTypeSizeT, &ffi.TypePointer)
	benchmarkInfoGetPrefillTokensPerSecAtFunc = newLazyFun(
		"litert_lm_benchmark_info_get_prefill_tokens_per_sec_at",
		&ffi.TypeSint32, &ffi.TypePointer, &ffiTypeSizeT, &ffi.TypePointer)
	benchmarkInfoGetDecodeTokensPerSecAtFunc = newLazyFun(
		"litert_lm_benchmark_info_get_decode_tokens_per_sec_at",
		&ffi.TypeSint32, &ffi.TypePointer, &ffiTypeSizeT, &ffi.TypePointer)
	benchmarkInfoGetMaxNumTokensFunc = newLazyFun(
		"litert_lm_benchmark_info_get_max_num_tokens",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)

	// ---- Conversation ----
	conversationCreateFunc = newLazyFun(
		"litert_lm_conversation_create",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer)
	conversationDeleteFunc = newLazyFun(
		"litert_lm_conversation_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	conversationCloneFunc = newLazyFun(
		"litert_lm_conversation_clone",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	conversationSendMessageFunc = newLazyFun(
		"litert_lm_conversation_send_message",
		&ffi.TypeSint32,
		&ffi.TypePointer, // conversation
		&ffi.TypePointer, // message_json
		&ffi.TypePointer, // extra_context
		&ffi.TypePointer, // optional_args
		&ffi.TypePointer, // out_response
	)
	conversationSendMessageStreamFunc = newLazyFun(
		"litert_lm_conversation_send_message_stream",
		&ffi.TypeSint32,
		&ffi.TypePointer, // conversation
		&ffi.TypePointer, // message_json
		&ffi.TypePointer, // extra_context
		&ffi.TypePointer, // optional_args
		&ffi.TypePointer, // callback
		&ffi.TypePointer, // callback_data
	)
	conversationCancelProcessFunc = newLazyFun(
		"litert_lm_conversation_cancel_process",
		&ffi.TypeSint32, &ffi.TypePointer)
	conversationWaitUntilDoneFunc = newLazyFun(
		"litert_lm_conversation_wait_until_done",
		&ffi.TypeSint32, &ffi.TypePointer)
	conversationRenderMessageToStringFunc = newLazyFun(
		"litert_lm_conversation_render_message_to_string",
		&ffi.TypeSint32,
		&ffi.TypePointer, // conversation
		&ffi.TypePointer, // message_json
		&ffi.TypePointer, // out_text
	)
	conversationRenderPrefaceToStringFunc = newLazyFun(
		"litert_lm_conversation_render_preface_to_string",
		&ffi.TypeSint32,
		&ffi.TypePointer, // conversation
		&ffi.TypePointer, // out_text
	)
	conversationGetBenchmarkInfoFunc = newLazyFun(
		"litert_lm_conversation_get_benchmark_info",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	conversationGetTokenCountFunc = newLazyFun(
		"litert_lm_conversation_get_token_count",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)

	// ---- JSON response ----
	jsonResponseDeleteFunc = newLazyFun(
		"litert_lm_json_response_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	jsonResponseGetStringFunc = newLazyFun(
		"litert_lm_json_response_get_string",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)

	// ---- Tokenizer ----
	engineTokenizeFunc = newLazyFun(
		"litert_lm_engine_tokenize",
		&ffi.TypeSint32,
		&ffi.TypePointer, // engine
		&ffi.TypePointer, // text (const char*)
		&ffi.TypePointer, // out_result
	)
	tokenizeResultDeleteFunc = newLazyFun(
		"litert_lm_tokenize_result_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	tokenizeResultGetTokensFunc = newLazyFun(
		"litert_lm_tokenize_result_get_tokens",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	tokenizeResultGetNumTokensFunc = newLazyFun(
		"litert_lm_tokenize_result_get_num_tokens",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	engineDetokenizeFunc = newLazyFun(
		"litert_lm_engine_detokenize",
		&ffi.TypeSint32,
		&ffi.TypePointer, // engine
		&ffi.TypePointer, // tokens (const int*)
		&ffiTypeSizeT,    // num_tokens
		&ffi.TypePointer, // out_result
	)
	detokenizeResultDeleteFunc = newLazyFun(
		"litert_lm_detokenize_result_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	detokenizeResultGetStringFunc = newLazyFun(
		"litert_lm_detokenize_result_get_string",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)

	// ---- Token unions / start+stop tokens ----
	engineGetStartTokenFunc = newLazyFun(
		"litert_lm_engine_get_start_token",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	engineGetStopTokensFunc = newLazyFun(
		"litert_lm_engine_get_stop_tokens",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	tokenUnionDeleteFunc = newLazyFun(
		"litert_lm_token_union_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	tokenUnionGetTypeFunc = newLazyFun(
		"litert_lm_token_union_get_type",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	tokenUnionGetStringFunc = newLazyFun(
		"litert_lm_token_union_get_string",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	tokenUnionGetIdsFunc = newLazyFun(
		"litert_lm_token_union_get_ids",
		&ffi.TypeSint32,
		&ffi.TypePointer, // token_union
		&ffi.TypePointer, // out_tokens (**int)
		&ffi.TypePointer, // out_num_tokens (*size_t)
	)
	tokenUnionsDeleteFunc = newLazyFun(
		"litert_lm_token_unions_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	tokenUnionsGetNumTokensFunc = newLazyFun(
		"litert_lm_token_unions_get_num_tokens",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypePointer)
	tokenUnionsGetTokenAtFunc = newLazyFun(
		"litert_lm_token_unions_get_token_at",
		&ffi.TypeSint32,
		&ffi.TypePointer, // tokens
		&ffiTypeSizeT,    // index
		&ffi.TypePointer, // out_token
	)

	// ---- Model capabilities / LoadedFile ----
	loadedFileCreateFunc = newLazyFun(
		"litert_lm_loaded_file_create",
		&ffi.TypeSint32,
		&ffi.TypePointer, // litertlm_path (const char*)
		&ffi.TypePointer, // out_loaded_file
	)
	loadedFileDeleteFunc = newLazyFun(
		"litert_lm_loaded_file_delete",
		&ffi.TypeVoid,
		&ffi.TypePointer, // loaded_file
	)
	loadedFileHasSpeculativeDecodingSupportFunc = newLazyFun(
		"litert_lm_loaded_file_has_speculative_decoding_support",
		&ffi.TypeSint32,
		&ffi.TypePointer, // loaded_file
		&ffi.TypePointer, // out_supported
	)
	loadedFileSupportsThinkingFunc = newLazyFun(
		"litert_lm_loaded_file_supports_thinking",
		&ffi.TypeSint32,
		&ffi.TypePointer, // loaded_file
		&ffi.TypePointer, // out_supported
	)
	loadedFileSupportsFunctionCallingFunc = newLazyFun(
		"litert_lm_loaded_file_supports_function_calling",
		&ffi.TypeSint32,
		&ffi.TypePointer, // loaded_file
		&ffi.TypePointer, // out_supported
	)
	loadedFileSamplerTypeFunc = newLazyFun(
		"litert_lm_loaded_file_sampler_type",
		&ffi.TypeSint32,
		&ffi.TypePointer, // loaded_file
		&ffi.TypePointer, // out_sampler_type
	)
	loadedFileSamplerTemperatureFunc = newLazyFun(
		"litert_lm_loaded_file_sampler_temperature",
		&ffi.TypeSint32,
		&ffi.TypePointer, // loaded_file
		&ffi.TypePointer, // out_temperature
	)
	loadedFileSamplerTopKFunc = newLazyFun(
		"litert_lm_loaded_file_sampler_top_k",
		&ffi.TypeSint32,
		&ffi.TypePointer, // loaded_file
		&ffi.TypePointer, // out_top_k
	)
	loadedFileSamplerTopPFunc = newLazyFun(
		"litert_lm_loaded_file_sampler_top_p",
		&ffi.TypeSint32,
		&ffi.TypePointer, // loaded_file
		&ffi.TypePointer, // out_top_p
	)
	loadedFileSupportsInputModalityFunc = newLazyFun(
		"litert_lm_loaded_file_supports_input_modality",
		&ffi.TypeSint32,
		&ffi.TypePointer, // loaded_file
		&ffi.TypeSint32,  // modality (LiteRtLmModality)
		&ffi.TypePointer, // out_supported
	)
	loadedFileMaxVisionTokenBudgetFunc = newLazyFun(
		"litert_lm_loaded_file_max_vision_token_budget",
		&ffi.TypeSint32,
		&ffi.TypePointer, // loaded_file
		&ffi.TypePointer, // out_budget
	)
)

// loadFuncs stashes the opened library so lazyFun can resolve symbols
// against it on first call. Symbol lookup itself happens lazily —
// missing symbols return StatusUnimplemented errors when their wrapper is
// actually invoked.
func loadFuncs(lib ffi.Lib) error {
	rt.setMainLib(lib, true)
	return nil
}
