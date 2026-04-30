package litertlm

import "github.com/jupiterrider/ffi"

// ffiTypeSizeT aliases the ffi type used for C `size_t`. On all supported
// platforms this is a 64-bit unsigned integer.
var ffiTypeSizeT = ffi.TypeUint64

// Every exported litert_lm_* function in c/engine.h has a package-level
// ffi.Fun variable here. New symbols need both a Fun and a lib.Prep
// in loadFuncs().
var (
	// Session config
	sessionConfigCreateFunc             ffi.Fun
	sessionConfigSetMaxOutputTokensFunc ffi.Fun
	sessionConfigSetSamplerParamsFunc   ffi.Fun
	sessionConfigDeleteFunc             ffi.Fun

	// Conversation config
	conversationConfigCreateFunc                       ffi.Fun
	conversationConfigSetSessionConfigFunc             ffi.Fun
	conversationConfigSetSystemMessageFunc             ffi.Fun
	conversationConfigSetToolsFunc                     ffi.Fun
	conversationConfigSetMessagesFunc                  ffi.Fun
	conversationConfigSetEnableConstrainedDecodingFunc ffi.Fun
	conversationConfigDeleteFunc                       ffi.Fun

	// Logging
	setMinLogLevelFunc ffi.Fun

	// Engine settings
	engineSettingsCreateFunc                        ffi.Fun
	engineSettingsDeleteFunc                        ffi.Fun
	engineSettingsSetMaxNumTokensFunc               ffi.Fun
	engineSettingsSetParallelFileSectionLoadingFunc ffi.Fun
	engineSettingsSetCacheDirFunc                   ffi.Fun
	engineSettingsSetActivationDataTypeFunc         ffi.Fun
	engineSettingsSetPrefillChunkSizeFunc           ffi.Fun
	engineSettingsEnableBenchmarkFunc               ffi.Fun
	engineSettingsSetNumPrefillTokensFunc           ffi.Fun
	engineSettingsSetNumDecodeTokensFunc            ffi.Fun

	// Engine
	engineCreateFunc        ffi.Fun
	engineDeleteFunc        ffi.Fun
	engineCreateSessionFunc ffi.Fun

	// Session
	sessionDeleteFunc                ffi.Fun
	sessionGenerateContentFunc       ffi.Fun
	sessionGenerateContentStreamFunc ffi.Fun
	sessionGetBenchmarkInfoFunc      ffi.Fun
	sessionCancelProcessFunc         ffi.Fun
	sessionRunPrefillFunc            ffi.Fun
	sessionRunDecodeFunc             ffi.Fun
	sessionRunTextScoringFunc        ffi.Fun

	// Responses
	responsesDeleteFunc            ffi.Fun
	responsesGetNumCandidatesFunc  ffi.Fun
	responsesGetResponseTextAtFunc ffi.Fun
	responsesHasScoreAtFunc        ffi.Fun
	responsesGetScoreAtFunc        ffi.Fun
	responsesHasTokenLengthAtFunc  ffi.Fun
	responsesGetTokenLengthAtFunc  ffi.Fun

	// Benchmark info
	benchmarkInfoDeleteFunc                   ffi.Fun
	benchmarkInfoGetTimeToFirstTokenFunc      ffi.Fun
	benchmarkInfoGetTotalInitTimeInSecondFunc ffi.Fun
	benchmarkInfoGetNumPrefillTurnsFunc       ffi.Fun
	benchmarkInfoGetNumDecodeTurnsFunc        ffi.Fun
	benchmarkInfoGetPrefillTokenCountAtFunc   ffi.Fun
	benchmarkInfoGetDecodeTokenCountAtFunc    ffi.Fun
	benchmarkInfoGetPrefillTokensPerSecAtFunc ffi.Fun
	benchmarkInfoGetDecodeTokensPerSecAtFunc  ffi.Fun

	// Conversation
	conversationCreateFunc            ffi.Fun
	conversationDeleteFunc            ffi.Fun
	conversationSendMessageFunc       ffi.Fun
	conversationSendMessageStreamFunc ffi.Fun
	conversationCancelProcessFunc     ffi.Fun
	conversationGetBenchmarkInfoFunc  ffi.Fun

	// JSON response
	jsonResponseDeleteFunc    ffi.Fun
	jsonResponseGetStringFunc ffi.Fun

	// Tokenizer
	engineTokenizeFunc             ffi.Fun
	tokenizeResultDeleteFunc       ffi.Fun
	tokenizeResultGetTokensFunc    ffi.Fun
	tokenizeResultGetNumTokensFunc ffi.Fun
	engineDetokenizeFunc           ffi.Fun
	detokenizeResultDeleteFunc     ffi.Fun
	detokenizeResultGetStringFunc  ffi.Fun

	// Token unions / start+stop tokens
	engineGetStartTokenFunc     ffi.Fun
	engineGetStopTokensFunc     ffi.Fun
	tokenUnionDeleteFunc        ffi.Fun
	tokenUnionGetTypeFunc       ffi.Fun
	tokenUnionGetStringFunc     ffi.Fun
	tokenUnionGetIdsFunc        ffi.Fun
	tokenUnionsDeleteFunc       ffi.Fun
	tokenUnionsGetNumTokensFunc ffi.Fun
	tokenUnionsGetTokenAtFunc   ffi.Fun
)

// loadFuncs registers every C entry point with the opened main library.
// Each Prep declares the return type first, then argument types in C
// declaration order.
func loadFuncs(lib ffi.Lib) error {
	var err error

	// ---- Session config ----
	if sessionConfigCreateFunc, err = lib.Prep(
		"litert_lm_session_config_create", &ffi.TypePointer); err != nil {
		return loadError("litert_lm_session_config_create", err)
	}
	if sessionConfigSetMaxOutputTokensFunc, err = lib.Prep(
		"litert_lm_session_config_set_max_output_tokens",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("litert_lm_session_config_set_max_output_tokens", err)
	}
	if sessionConfigSetSamplerParamsFunc, err = lib.Prep(
		"litert_lm_session_config_set_sampler_params",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_session_config_set_sampler_params", err)
	}
	if sessionConfigDeleteFunc, err = lib.Prep(
		"litert_lm_session_config_delete",
		&ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_session_config_delete", err)
	}

	// ---- Conversation config ----
	// create() takes no args; fields are populated via per-field setters.
	if conversationConfigCreateFunc, err = lib.Prep(
		"litert_lm_conversation_config_create",
		&ffi.TypePointer,
	); err != nil {
		return loadError("litert_lm_conversation_config_create", err)
	}
	if conversationConfigSetSessionConfigFunc, err = lib.Prep(
		"litert_lm_conversation_config_set_session_config",
		&ffi.TypeVoid,
		&ffi.TypePointer, // config
		&ffi.TypePointer, // session_config
	); err != nil {
		return loadError("litert_lm_conversation_config_set_session_config", err)
	}
	if conversationConfigSetSystemMessageFunc, err = lib.Prep(
		"litert_lm_conversation_config_set_system_message",
		&ffi.TypeVoid,
		&ffi.TypePointer, // config
		&ffi.TypePointer, // system_message_json (const char*)
	); err != nil {
		return loadError("litert_lm_conversation_config_set_system_message", err)
	}
	if conversationConfigSetToolsFunc, err = lib.Prep(
		"litert_lm_conversation_config_set_tools",
		&ffi.TypeVoid,
		&ffi.TypePointer, // config
		&ffi.TypePointer, // tools_json (const char*)
	); err != nil {
		return loadError("litert_lm_conversation_config_set_tools", err)
	}
	if conversationConfigSetMessagesFunc, err = lib.Prep(
		"litert_lm_conversation_config_set_messages",
		&ffi.TypeVoid,
		&ffi.TypePointer, // config
		&ffi.TypePointer, // messages_json (const char*)
	); err != nil {
		return loadError("litert_lm_conversation_config_set_messages", err)
	}
	if conversationConfigSetEnableConstrainedDecodingFunc, err = lib.Prep(
		"litert_lm_conversation_config_set_enable_constrained_decoding",
		&ffi.TypeVoid,
		&ffi.TypePointer, // config
		&ffi.TypeUint8,   // enable_constrained_decoding (bool)
	); err != nil {
		return loadError("litert_lm_conversation_config_set_enable_constrained_decoding", err)
	}
	if conversationConfigDeleteFunc, err = lib.Prep(
		"litert_lm_conversation_config_delete",
		&ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_conversation_config_delete", err)
	}

	// ---- Logging ----
	if setMinLogLevelFunc, err = lib.Prep(
		"litert_lm_set_min_log_level",
		&ffi.TypeVoid, &ffi.TypeSint32); err != nil {
		return loadError("litert_lm_set_min_log_level", err)
	}

	// ---- Engine settings ----
	if engineSettingsCreateFunc, err = lib.Prep(
		"litert_lm_engine_settings_create",
		&ffi.TypePointer,
		&ffi.TypePointer, // model_path
		&ffi.TypePointer, // backend
		&ffi.TypePointer, // vision_backend (nullable)
		&ffi.TypePointer, // audio_backend (nullable)
	); err != nil {
		return loadError("litert_lm_engine_settings_create", err)
	}
	if engineSettingsDeleteFunc, err = lib.Prep(
		"litert_lm_engine_settings_delete",
		&ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_engine_settings_delete", err)
	}
	if engineSettingsSetMaxNumTokensFunc, err = lib.Prep(
		"litert_lm_engine_settings_set_max_num_tokens",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("litert_lm_engine_settings_set_max_num_tokens", err)
	}
	if engineSettingsSetParallelFileSectionLoadingFunc, err = lib.Prep(
		"litert_lm_engine_settings_set_parallel_file_section_loading",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeUint8); err != nil {
		return loadError("litert_lm_engine_settings_set_parallel_file_section_loading", err)
	}
	if engineSettingsSetCacheDirFunc, err = lib.Prep(
		"litert_lm_engine_settings_set_cache_dir",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_engine_settings_set_cache_dir", err)
	}
	if engineSettingsSetActivationDataTypeFunc, err = lib.Prep(
		"litert_lm_engine_settings_set_activation_data_type",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("litert_lm_engine_settings_set_activation_data_type", err)
	}
	if engineSettingsSetPrefillChunkSizeFunc, err = lib.Prep(
		"litert_lm_engine_settings_set_prefill_chunk_size",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("litert_lm_engine_settings_set_prefill_chunk_size", err)
	}
	if engineSettingsEnableBenchmarkFunc, err = lib.Prep(
		"litert_lm_engine_settings_enable_benchmark",
		&ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_engine_settings_enable_benchmark", err)
	}
	if engineSettingsSetNumPrefillTokensFunc, err = lib.Prep(
		"litert_lm_engine_settings_set_num_prefill_tokens",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("litert_lm_engine_settings_set_num_prefill_tokens", err)
	}
	if engineSettingsSetNumDecodeTokensFunc, err = lib.Prep(
		"litert_lm_engine_settings_set_num_decode_tokens",
		&ffi.TypeVoid, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("litert_lm_engine_settings_set_num_decode_tokens", err)
	}

	// ---- Engine ----
	if engineCreateFunc, err = lib.Prep(
		"litert_lm_engine_create",
		&ffi.TypePointer, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_engine_create", err)
	}
	if engineDeleteFunc, err = lib.Prep(
		"litert_lm_engine_delete",
		&ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_engine_delete", err)
	}
	if engineCreateSessionFunc, err = lib.Prep(
		"litert_lm_engine_create_session",
		&ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_engine_create_session", err)
	}

	// ---- Session ----
	if sessionDeleteFunc, err = lib.Prep(
		"litert_lm_session_delete",
		&ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_session_delete", err)
	}
	if sessionGenerateContentFunc, err = lib.Prep(
		"litert_lm_session_generate_content",
		&ffi.TypePointer,
		&ffi.TypePointer, // session
		&ffi.TypePointer, // inputs
		&ffiTypeSizeT,    // num_inputs
	); err != nil {
		return loadError("litert_lm_session_generate_content", err)
	}
	if sessionGenerateContentStreamFunc, err = lib.Prep(
		"litert_lm_session_generate_content_stream",
		&ffi.TypeSint32,
		&ffi.TypePointer, // session
		&ffi.TypePointer, // inputs
		&ffiTypeSizeT,    // num_inputs
		&ffi.TypePointer, // callback
		&ffi.TypePointer, // callback_data
	); err != nil {
		return loadError("litert_lm_session_generate_content_stream", err)
	}
	if sessionGetBenchmarkInfoFunc, err = lib.Prep(
		"litert_lm_session_get_benchmark_info",
		&ffi.TypePointer, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_session_get_benchmark_info", err)
	}
	if sessionCancelProcessFunc, err = lib.Prep(
		"litert_lm_session_cancel_process",
		&ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_session_cancel_process", err)
	}
	if sessionRunPrefillFunc, err = lib.Prep(
		"litert_lm_session_run_prefill",
		&ffi.TypeSint32,
		&ffi.TypePointer, // session
		&ffi.TypePointer, // inputs
		&ffiTypeSizeT,    // num_inputs
	); err != nil {
		return loadError("litert_lm_session_run_prefill", err)
	}
	if sessionRunDecodeFunc, err = lib.Prep(
		"litert_lm_session_run_decode",
		&ffi.TypePointer, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_session_run_decode", err)
	}
	if sessionRunTextScoringFunc, err = lib.Prep(
		"litert_lm_session_run_text_scoring",
		&ffi.TypePointer,
		&ffi.TypePointer, // session
		&ffi.TypePointer, // target_text (const char**)
		&ffiTypeSizeT,    // num_targets
		&ffi.TypeUint8,   // store_token_lengths
	); err != nil {
		return loadError("litert_lm_session_run_text_scoring", err)
	}

	// ---- Responses ----
	if responsesDeleteFunc, err = lib.Prep(
		"litert_lm_responses_delete",
		&ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_responses_delete", err)
	}
	if responsesGetNumCandidatesFunc, err = lib.Prep(
		"litert_lm_responses_get_num_candidates",
		&ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_responses_get_num_candidates", err)
	}
	if responsesGetResponseTextAtFunc, err = lib.Prep(
		"litert_lm_responses_get_response_text_at",
		&ffi.TypePointer, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("litert_lm_responses_get_response_text_at", err)
	}
	if responsesHasScoreAtFunc, err = lib.Prep(
		"litert_lm_responses_has_score_at",
		&ffi.TypeUint8, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("litert_lm_responses_has_score_at", err)
	}
	if responsesGetScoreAtFunc, err = lib.Prep(
		"litert_lm_responses_get_score_at",
		&ffi.TypeFloat, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("litert_lm_responses_get_score_at", err)
	}
	if responsesHasTokenLengthAtFunc, err = lib.Prep(
		"litert_lm_responses_has_token_length_at",
		&ffi.TypeUint8, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("litert_lm_responses_has_token_length_at", err)
	}
	if responsesGetTokenLengthAtFunc, err = lib.Prep(
		"litert_lm_responses_get_token_length_at",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("litert_lm_responses_get_token_length_at", err)
	}

	// ---- Benchmark info ----
	if benchmarkInfoDeleteFunc, err = lib.Prep(
		"litert_lm_benchmark_info_delete",
		&ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_benchmark_info_delete", err)
	}
	if benchmarkInfoGetTimeToFirstTokenFunc, err = lib.Prep(
		"litert_lm_benchmark_info_get_time_to_first_token",
		&ffi.TypeDouble, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_benchmark_info_get_time_to_first_token", err)
	}
	if benchmarkInfoGetTotalInitTimeInSecondFunc, err = lib.Prep(
		"litert_lm_benchmark_info_get_total_init_time_in_second",
		&ffi.TypeDouble, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_benchmark_info_get_total_init_time_in_second", err)
	}
	if benchmarkInfoGetNumPrefillTurnsFunc, err = lib.Prep(
		"litert_lm_benchmark_info_get_num_prefill_turns",
		&ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_benchmark_info_get_num_prefill_turns", err)
	}
	if benchmarkInfoGetNumDecodeTurnsFunc, err = lib.Prep(
		"litert_lm_benchmark_info_get_num_decode_turns",
		&ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_benchmark_info_get_num_decode_turns", err)
	}
	if benchmarkInfoGetPrefillTokenCountAtFunc, err = lib.Prep(
		"litert_lm_benchmark_info_get_prefill_token_count_at",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("litert_lm_benchmark_info_get_prefill_token_count_at", err)
	}
	if benchmarkInfoGetDecodeTokenCountAtFunc, err = lib.Prep(
		"litert_lm_benchmark_info_get_decode_token_count_at",
		&ffi.TypeSint32, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("litert_lm_benchmark_info_get_decode_token_count_at", err)
	}
	if benchmarkInfoGetPrefillTokensPerSecAtFunc, err = lib.Prep(
		"litert_lm_benchmark_info_get_prefill_tokens_per_sec_at",
		&ffi.TypeDouble, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("litert_lm_benchmark_info_get_prefill_tokens_per_sec_at", err)
	}
	if benchmarkInfoGetDecodeTokensPerSecAtFunc, err = lib.Prep(
		"litert_lm_benchmark_info_get_decode_tokens_per_sec_at",
		&ffi.TypeDouble, &ffi.TypePointer, &ffi.TypeSint32); err != nil {
		return loadError("litert_lm_benchmark_info_get_decode_tokens_per_sec_at", err)
	}

	// ---- Conversation ----
	if conversationCreateFunc, err = lib.Prep(
		"litert_lm_conversation_create",
		&ffi.TypePointer, &ffi.TypePointer, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_conversation_create", err)
	}
	if conversationDeleteFunc, err = lib.Prep(
		"litert_lm_conversation_delete",
		&ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_conversation_delete", err)
	}
	if conversationSendMessageFunc, err = lib.Prep(
		"litert_lm_conversation_send_message",
		&ffi.TypePointer,
		&ffi.TypePointer, // conversation
		&ffi.TypePointer, // message_json
		&ffi.TypePointer, // extra_context
	); err != nil {
		return loadError("litert_lm_conversation_send_message", err)
	}
	if conversationSendMessageStreamFunc, err = lib.Prep(
		"litert_lm_conversation_send_message_stream",
		&ffi.TypeSint32,
		&ffi.TypePointer, // conversation
		&ffi.TypePointer, // message_json
		&ffi.TypePointer, // extra_context
		&ffi.TypePointer, // callback
		&ffi.TypePointer, // callback_data
	); err != nil {
		return loadError("litert_lm_conversation_send_message_stream", err)
	}
	if conversationCancelProcessFunc, err = lib.Prep(
		"litert_lm_conversation_cancel_process",
		&ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_conversation_cancel_process", err)
	}
	if conversationGetBenchmarkInfoFunc, err = lib.Prep(
		"litert_lm_conversation_get_benchmark_info",
		&ffi.TypePointer, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_conversation_get_benchmark_info", err)
	}

	// ---- JSON response ----
	if jsonResponseDeleteFunc, err = lib.Prep(
		"litert_lm_json_response_delete",
		&ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_json_response_delete", err)
	}
	if jsonResponseGetStringFunc, err = lib.Prep(
		"litert_lm_json_response_get_string",
		&ffi.TypePointer, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_json_response_get_string", err)
	}

	// ---- Tokenizer ----
	if engineTokenizeFunc, err = lib.Prep(
		"litert_lm_engine_tokenize",
		&ffi.TypePointer,
		&ffi.TypePointer, // engine
		&ffi.TypePointer, // text (const char*)
	); err != nil {
		return loadError("litert_lm_engine_tokenize", err)
	}
	if tokenizeResultDeleteFunc, err = lib.Prep(
		"litert_lm_tokenize_result_delete",
		&ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_tokenize_result_delete", err)
	}
	if tokenizeResultGetTokensFunc, err = lib.Prep(
		"litert_lm_tokenize_result_get_tokens",
		&ffi.TypePointer, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_tokenize_result_get_tokens", err)
	}
	if tokenizeResultGetNumTokensFunc, err = lib.Prep(
		"litert_lm_tokenize_result_get_num_tokens",
		&ffiTypeSizeT, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_tokenize_result_get_num_tokens", err)
	}
	if engineDetokenizeFunc, err = lib.Prep(
		"litert_lm_engine_detokenize",
		&ffi.TypePointer,
		&ffi.TypePointer, // engine
		&ffi.TypePointer, // tokens (const int*)
		&ffiTypeSizeT,    // num_tokens
	); err != nil {
		return loadError("litert_lm_engine_detokenize", err)
	}
	if detokenizeResultDeleteFunc, err = lib.Prep(
		"litert_lm_detokenize_result_delete",
		&ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_detokenize_result_delete", err)
	}
	if detokenizeResultGetStringFunc, err = lib.Prep(
		"litert_lm_detokenize_result_get_string",
		&ffi.TypePointer, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_detokenize_result_get_string", err)
	}

	// ---- Token unions / start+stop tokens ----
	if engineGetStartTokenFunc, err = lib.Prep(
		"litert_lm_engine_get_start_token",
		&ffi.TypePointer, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_engine_get_start_token", err)
	}
	if engineGetStopTokensFunc, err = lib.Prep(
		"litert_lm_engine_get_stop_tokens",
		&ffi.TypePointer, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_engine_get_stop_tokens", err)
	}
	if tokenUnionDeleteFunc, err = lib.Prep(
		"litert_lm_token_union_delete",
		&ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_token_union_delete", err)
	}
	if tokenUnionGetTypeFunc, err = lib.Prep(
		"litert_lm_token_union_get_type",
		&ffi.TypeSint32, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_token_union_get_type", err)
	}
	if tokenUnionGetStringFunc, err = lib.Prep(
		"litert_lm_token_union_get_string",
		&ffi.TypePointer, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_token_union_get_string", err)
	}
	if tokenUnionGetIdsFunc, err = lib.Prep(
		"litert_lm_token_union_get_ids",
		&ffi.TypeSint32,
		&ffi.TypePointer, // token_union
		&ffi.TypePointer, // out_tokens (**int)
		&ffi.TypePointer, // out_num_tokens (*size_t)
	); err != nil {
		return loadError("litert_lm_token_union_get_ids", err)
	}
	if tokenUnionsDeleteFunc, err = lib.Prep(
		"litert_lm_token_unions_delete",
		&ffi.TypeVoid, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_token_unions_delete", err)
	}
	if tokenUnionsGetNumTokensFunc, err = lib.Prep(
		"litert_lm_token_unions_get_num_tokens",
		&ffiTypeSizeT, &ffi.TypePointer); err != nil {
		return loadError("litert_lm_token_unions_get_num_tokens", err)
	}
	if tokenUnionsGetTokenAtFunc, err = lib.Prep(
		"litert_lm_token_unions_get_token_at",
		&ffi.TypePointer,
		&ffi.TypePointer, // tokens
		&ffiTypeSizeT,    // index
	); err != nil {
		return loadError("litert_lm_token_unions_get_token_at", err)
	}

	return nil
}
