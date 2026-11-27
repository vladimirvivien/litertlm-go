package litertlm

// Handle types. Each wraps an opaque C pointer as a uintptr. The zero value
// is null and may be tested with an ordinary `if h == 0` check.
//
// The types deliberately mirror the opaque struct names declared in
// LiteRT-LM's c/engine.h so a reader moving between the C and Go layers can
// follow the correspondence by sight.

// Engine wraps a LiteRtLmEngine* and owns a loaded model.
type Engine uintptr

// EngineSettings wraps a LiteRtLmEngineSettings* — the configuration used to
// create an Engine.
type EngineSettings uintptr

// Session wraps a LiteRtLmSession* — per-turn generation state rooted in an
// Engine.
type Session uintptr

// SessionConfig wraps a LiteRtLmSessionConfig* — session-level configuration
// (sampler params, max output tokens).
type SessionConfig uintptr

// Responses wraps a LiteRtLmResponses* returned by synchronous generation.
type Responses uintptr

// BenchmarkInfo wraps a LiteRtLmBenchmarkInfo* — per-session / per-conversation
// timing statistics.
type BenchmarkInfo uintptr

// Conversation wraps a LiteRtLmConversation* — the higher-level multi-turn
// chat interface.
type Conversation uintptr

// ConversationConfig wraps a LiteRtLmConversationConfig*.
type ConversationConfig uintptr

// OptionalArgs wraps a LiteRtLmConversationOptionalArgs* — per-turn
// knobs passed to Conversation.SendMessage / SendMessageStream.
type OptionalArgs uintptr

// JSONResponse wraps a LiteRtLmJsonResponse* returned by Conversation calls.
type JSONResponse uintptr

// InputData wraps a LiteRtLmInputData* opaque pointer representing input content.
type InputData uintptr

// LoadedFile wraps a LiteRtLmLoadedFile* opaque pointer representing a model
// file opened for capability inspection.
type LoadedFile uintptr

// EmbeddingEngine wraps a LiteRtLmEmbeddingEngine* for multimodal and text
// embedding generation.
type EmbeddingEngine uintptr

// EmbeddingEngineSettings wraps a LiteRtLmEmbeddingEngineSettings* used to
// configure an EmbeddingEngine.
type EmbeddingEngineSettings uintptr

// EmbeddingOptions wraps a LiteRtLmEmbeddingOptions* carrying per-call
// normalization, special-token, overflow, truncation, and vision settings.
type EmbeddingOptions uintptr

// EmbeddingResponse wraps a LiteRtLmEmbeddingResponse* holding one embedding
// vector.
type EmbeddingResponse uintptr

// EmbeddingResponses wraps a LiteRtLmEmbeddingResponses* holding a batch of
// embedding vectors.
type EmbeddingResponses uintptr

// ModelSyncOptionsHandle wraps a LiteRtLmModelSyncOptions* configuring
// differential model synchronization.
type ModelSyncOptionsHandle uintptr

// ModelSyncPlanHandle wraps a LiteRtLmModelSyncPlan* holding a computed
// synchronization plan comparing local and remote models.
type ModelSyncPlanHandle uintptr

// ModelSyncRangeWriter wraps a borrowed LiteRtLmModelSyncRangeWriter* passed to
// a RangeFetcher callback during synchronization. Valid only during the call.
type ModelSyncRangeWriter uintptr

// ModelSyncResultHandle wraps a LiteRtLmModelSyncResult* returned by cache sync.
type ModelSyncResultHandle uintptr

// SessionDebugInfo wraps a LiteRtLmSessionDebugInfo* holding debugger capture
// metadata for a Session or Conversation.
type SessionDebugInfo uintptr
