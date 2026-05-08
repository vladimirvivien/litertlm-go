package litertlm

import "github.com/jupiterrider/ffi"

// ffiTypeSizeT aliases the ffi type used for C `size_t`. On all supported
// platforms this is a 64-bit unsigned integer.
var ffiTypeSizeT = ffi.TypeUint64

// Every exported litert_lm_* function in c/engine.h has a package-level
// *lazyFun declared here. Each one resolves its C symbol against
// libHandle on first Call, panicking with the symbol name if the
// loaded library doesn't export it. Adding a new binding is a single
// newLazyFun(...) line — no loadFuncs entry, no Load-time failure if
// the user's library predates the symbol.

var (
	// ---- Session config ----
	sessionConfigCreateFunc = newLazyFun(
		"litert_lm_session_config_create",
		&ffi.TypePointer)
	sessionConfigSetMaxOutputTokensFunc = newLazyFun(
		"litert_lm_session_config_set_max_output_tokens",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeSint32)
	sessionConfigSetSamplerParamsFunc = newLazyFun(
		"litert_lm_session_config_set_sampler_params",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypePointer)
	sessionConfigSetApplyPromptTemplateFunc = newLazyFun(
		"litert_lm_session_config_set_apply_prompt_template",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeUint8)
	sessionConfigDeleteFunc = newLazyFun(
		"litert_lm_session_config_delete",
		&ffi.TypeVoid, &ffi.TypePointer)

	// ---- Conversation config ----
	// create() takes no args; fields are populated via per-field setters.
	conversationConfigCreateFunc = newLazyFun(
		"litert_lm_conversation_config_create",
		&ffi.TypePointer)
	conversationConfigSetSessionConfigFunc = newLazyFun(
		"litert_lm_conversation_config_set_session_config",
		&ffi.TypeVoid,
		&ffi.TypePointer, // config
		&ffi.TypePointer, // session_config
	)
	conversationConfigSetSystemMessageFunc = newLazyFun(
		"litert_lm_conversation_config_set_system_message",
		&ffi.TypeVoid,
		&ffi.TypePointer, // config
		&ffi.TypePointer, // system_message_json (const char*)
	)
	conversationConfigSetToolsFunc = newLazyFun(
		"litert_lm_conversation_config_set_tools",
		&ffi.TypeVoid,
		&ffi.TypePointer, // config
		&ffi.TypePointer, // tools_json (const char*)
	)
	conversationConfigSetMessagesFunc = newLazyFun(
		"litert_lm_conversation_config_set_messages",
		&ffi.TypeVoid,
		&ffi.TypePointer, // config
		&ffi.TypePointer, // messages_json (const char*)
	)
	conversationConfigSetEnableConstrainedDecodingFunc = newLazyFun(
		"litert_lm_conversation_config_set_enable_constrained_decoding",
		&ffi.TypeVoid,
		&ffi.TypePointer, // config
		&ffi.TypeUint8,   // enable_constrained_decoding (bool)
	)
	conversationConfigSetExtraContextFunc = newLazyFun(
		"litert_lm_conversation_config_set_extra_context",
		&ffi.TypeVoid,
		&ffi.TypePointer, // config
		&ffi.TypePointer, // extra_context_json (const char*)
	)
	conversationConfigSetFilterChannelContentFromKVCacheFunc = newLazyFun(
		"litert_lm_conversation_config_set_filter_channel_content_from_kv_cache",
		&ffi.TypeVoid,
		&ffi.TypePointer, // config
		&ffi.TypeUint8,   // filter_channel_content_from_kv_cache (bool)
	)
	conversationConfigDeleteFunc = newLazyFun(
		"litert_lm_conversation_config_delete",
		&ffi.TypeVoid, &ffi.TypePointer)

	// ---- Logging ----
	setMinLogLevelFunc = newLazyFun(
		"litert_lm_set_min_log_level",
		&ffi.TypeVoid, &ffi.TypeSint32)

	// ---- Engine settings ----
	engineSettingsCreateFunc = newLazyFun(
		"litert_lm_engine_settings_create",
		&ffi.TypePointer,
		&ffi.TypePointer, // model_path
		&ffi.TypePointer, // backend
		&ffi.TypePointer, // vision_backend (nullable)
		&ffi.TypePointer, // audio_backend (nullable)
	)
	engineSettingsDeleteFunc = newLazyFun(
		"litert_lm_engine_settings_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	engineSettingsSetMaxNumTokensFunc = newLazyFun(
		"litert_lm_engine_settings_set_max_num_tokens",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeSint32)
	engineSettingsSetParallelFileSectionLoadingFunc = newLazyFun(
		"litert_lm_engine_settings_set_parallel_file_section_loading",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeUint8)
	engineSettingsSetCacheDirFunc = newLazyFun(
		"litert_lm_engine_settings_set_cache_dir",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypePointer)
	engineSettingsSetActivationDataTypeFunc = newLazyFun(
		"litert_lm_engine_settings_set_activation_data_type",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeSint32)
	engineSettingsSetPrefillChunkSizeFunc = newLazyFun(
		"litert_lm_engine_settings_set_prefill_chunk_size",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeSint32)
	engineSettingsEnableBenchmarkFunc = newLazyFun(
		"litert_lm_engine_settings_enable_benchmark",
		&ffi.TypeVoid, &ffi.TypePointer)
	engineSettingsSetNumPrefillTokensFunc = newLazyFun(
		"litert_lm_engine_settings_set_num_prefill_tokens",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeSint32)
	engineSettingsSetNumDecodeTokensFunc = newLazyFun(
		"litert_lm_engine_settings_set_num_decode_tokens",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeSint32)
	engineSettingsSetEnableSpeculativeDecodingFunc = newLazyFun(
		"litert_lm_engine_settings_set_enable_speculative_decoding",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeUint8)
	engineSettingsSetMaxNumImagesFunc = newLazyFun(
		"litert_lm_engine_settings_set_max_num_images",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeSint32)
	engineSettingsSetLitertDispatchLibDirFunc = newLazyFun(
		"litert_lm_engine_settings_set_litert_dispatch_lib_dir",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypePointer)

	// ---- Engine ----
	engineCreateFunc = newLazyFun(
		"litert_lm_engine_create",
		&ffi.TypePointer, &ffi.TypePointer)
	engineDeleteFunc = newLazyFun(
		"litert_lm_engine_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	engineCreateSessionFunc = newLazyFun(
		"litert_lm_engine_create_session",
		&ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer)

	// ---- Session ----
	sessionDeleteFunc = newLazyFun(
		"litert_lm_session_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	sessionGenerateContentFunc = newLazyFun(
		"litert_lm_session_generate_content",
		&ffi.TypePointer,
		&ffi.TypePointer, // session
		&ffi.TypePointer, // inputs
		&ffiTypeSizeT,    // num_inputs
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
		&ffi.TypePointer, &ffi.TypePointer)
	sessionCancelProcessFunc = newLazyFun(
		"litert_lm_session_cancel_process",
		&ffi.TypeVoid, &ffi.TypePointer)
	sessionRunPrefillFunc = newLazyFun(
		"litert_lm_session_run_prefill",
		&ffi.TypeSint32,
		&ffi.TypePointer, // session
		&ffi.TypePointer, // inputs
		&ffiTypeSizeT,    // num_inputs
	)
	sessionRunDecodeFunc = newLazyFun(
		"litert_lm_session_run_decode",
		&ffi.TypePointer, &ffi.TypePointer)
	sessionRunTextScoringFunc = newLazyFun(
		"litert_lm_session_run_text_scoring",
		&ffi.TypePointer,
		&ffi.TypePointer, // session
		&ffi.TypePointer, // target_text (const char**)
		&ffiTypeSizeT,    // num_targets
		&ffi.TypeUint8,   // store_token_lengths
	)

	// ---- Responses ----
	responsesDeleteFunc = newLazyFun(
		"litert_lm_responses_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	responsesGetNumCandidatesFunc = newLazyFun(
		"litert_lm_responses_get_num_candidates",
		&ffi.TypeSint32, &ffi.TypePointer)
	responsesGetResponseTextAtFunc = newLazyFun(
		"litert_lm_responses_get_response_text_at",
		&ffi.TypePointer, &ffi.TypePointer, &ffi.TypeSint32)
	responsesHasScoreAtFunc = newLazyFun(
		"litert_lm_responses_has_score_at",
		&ffi.TypeUint8, &ffi.TypePointer, &ffi.TypeSint32)
	responsesGetScoreAtFunc = newLazyFun(
		"litert_lm_responses_get_score_at",
		&ffi.TypeFloat, &ffi.TypePointer, &ffi.TypeSint32)
	responsesHasTokenLengthAtFunc = newLazyFun(
		"litert_lm_responses_has_token_length_at",
		&ffi.TypeUint8, &ffi.TypePointer, &ffi.TypeSint32)
	responsesHasTokenScoresAtFunc = newLazyFun(
		"litert_lm_responses_has_token_scores_at",
		&ffi.TypeUint8, &ffi.TypePointer, &ffi.TypeSint32)
	responsesGetNumTokenScoresAtFunc = newLazyFun(
		"litert_lm_responses_get_num_token_scores_at",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	responsesGetTokenScoresAtFunc = newLazyFun(
		"litert_lm_responses_get_token_scores_at",
		&ffi.TypePointer, &ffi.TypePointer, &ffi.TypeSint32)
	responsesGetTokenLengthAtFunc = newLazyFun(
		"litert_lm_responses_get_token_length_at",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)

	// ---- Benchmark info ----
	benchmarkInfoDeleteFunc = newLazyFun(
		"litert_lm_benchmark_info_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	benchmarkInfoGetTimeToFirstTokenFunc = newLazyFun(
		"litert_lm_benchmark_info_get_time_to_first_token",
		&ffi.TypeDouble, &ffi.TypePointer)
	benchmarkInfoGetTotalInitTimeInSecondFunc = newLazyFun(
		"litert_lm_benchmark_info_get_total_init_time_in_second",
		&ffi.TypeDouble, &ffi.TypePointer)
	benchmarkInfoGetNumPrefillTurnsFunc = newLazyFun(
		"litert_lm_benchmark_info_get_num_prefill_turns",
		&ffi.TypeSint32, &ffi.TypePointer)
	benchmarkInfoGetNumDecodeTurnsFunc = newLazyFun(
		"litert_lm_benchmark_info_get_num_decode_turns",
		&ffi.TypeSint32, &ffi.TypePointer)
	benchmarkInfoGetPrefillTokenCountAtFunc = newLazyFun(
		"litert_lm_benchmark_info_get_prefill_token_count_at",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	benchmarkInfoGetDecodeTokenCountAtFunc = newLazyFun(
		"litert_lm_benchmark_info_get_decode_token_count_at",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32)
	benchmarkInfoGetPrefillTokensPerSecAtFunc = newLazyFun(
		"litert_lm_benchmark_info_get_prefill_tokens_per_sec_at",
		&ffi.TypeDouble, &ffi.TypePointer, &ffi.TypeSint32)
	benchmarkInfoGetDecodeTokensPerSecAtFunc = newLazyFun(
		"litert_lm_benchmark_info_get_decode_tokens_per_sec_at",
		&ffi.TypeDouble, &ffi.TypePointer, &ffi.TypeSint32)

	// ---- Conversation ----
	conversationCreateFunc = newLazyFun(
		"litert_lm_conversation_create",
		&ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer)
	conversationDeleteFunc = newLazyFun(
		"litert_lm_conversation_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	conversationSendMessageFunc = newLazyFun(
		"litert_lm_conversation_send_message",
		&ffi.TypePointer,
		&ffi.TypePointer, // conversation
		&ffi.TypePointer, // message_json
		&ffi.TypePointer, // extra_context
	)
	conversationSendMessageStreamFunc = newLazyFun(
		"litert_lm_conversation_send_message_stream",
		&ffi.TypeSint32,
		&ffi.TypePointer, // conversation
		&ffi.TypePointer, // message_json
		&ffi.TypePointer, // extra_context
		&ffi.TypePointer, // callback
		&ffi.TypePointer, // callback_data
	)
	conversationCancelProcessFunc = newLazyFun(
		"litert_lm_conversation_cancel_process",
		&ffi.TypeVoid, &ffi.TypePointer)
	conversationRenderMessageToStringFunc = newLazyFun(
		"litert_lm_conversation_render_message_to_string",
		&ffi.TypePointer, // returned const char*
		&ffi.TypePointer, // conversation
		&ffi.TypePointer, // message_json
	)
	conversationGetBenchmarkInfoFunc = newLazyFun(
		"litert_lm_conversation_get_benchmark_info",
		&ffi.TypePointer, &ffi.TypePointer)

	// ---- JSON response ----
	jsonResponseDeleteFunc = newLazyFun(
		"litert_lm_json_response_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	jsonResponseGetStringFunc = newLazyFun(
		"litert_lm_json_response_get_string",
		&ffi.TypePointer, &ffi.TypePointer)

	// ---- Tokenizer ----
	engineTokenizeFunc = newLazyFun(
		"litert_lm_engine_tokenize",
		&ffi.TypePointer,
		&ffi.TypePointer, // engine
		&ffi.TypePointer, // text (const char*)
	)
	tokenizeResultDeleteFunc = newLazyFun(
		"litert_lm_tokenize_result_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	tokenizeResultGetTokensFunc = newLazyFun(
		"litert_lm_tokenize_result_get_tokens",
		&ffi.TypePointer, &ffi.TypePointer)
	tokenizeResultGetNumTokensFunc = newLazyFun(
		"litert_lm_tokenize_result_get_num_tokens",
		&ffiTypeSizeT, &ffi.TypePointer)
	engineDetokenizeFunc = newLazyFun(
		"litert_lm_engine_detokenize",
		&ffi.TypePointer,
		&ffi.TypePointer, // engine
		&ffi.TypePointer, // tokens (const int*)
		&ffiTypeSizeT,    // num_tokens
	)
	detokenizeResultDeleteFunc = newLazyFun(
		"litert_lm_detokenize_result_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	detokenizeResultGetStringFunc = newLazyFun(
		"litert_lm_detokenize_result_get_string",
		&ffi.TypePointer, &ffi.TypePointer)

	// ---- Token unions / start+stop tokens ----
	engineGetStartTokenFunc = newLazyFun(
		"litert_lm_engine_get_start_token",
		&ffi.TypePointer, &ffi.TypePointer)
	engineGetStopTokensFunc = newLazyFun(
		"litert_lm_engine_get_stop_tokens",
		&ffi.TypePointer, &ffi.TypePointer)
	tokenUnionDeleteFunc = newLazyFun(
		"litert_lm_token_union_delete",
		&ffi.TypeVoid, &ffi.TypePointer)
	tokenUnionGetTypeFunc = newLazyFun(
		"litert_lm_token_union_get_type",
		&ffi.TypeSint32, &ffi.TypePointer)
	tokenUnionGetStringFunc = newLazyFun(
		"litert_lm_token_union_get_string",
		&ffi.TypePointer, &ffi.TypePointer)
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
		&ffiTypeSizeT, &ffi.TypePointer)
	tokenUnionsGetTokenAtFunc = newLazyFun(
		"litert_lm_token_unions_get_token_at",
		&ffi.TypePointer,
		&ffi.TypePointer, // tokens
		&ffiTypeSizeT,    // index
	)
)

// loadFuncs stashes the opened library so lazyFun can resolve symbols
// against it on first call. Symbol lookup itself happens lazily —
// missing symbols don't fail Load, they panic (with the symbol name)
// only when their wrapper is actually invoked.
func loadFuncs(lib ffi.Lib) error {
	libHandle = lib
	return nil
}
