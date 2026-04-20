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

// JsonResponse wraps a LiteRtLmJsonResponse* returned by Conversation calls.
type JsonResponse uintptr
