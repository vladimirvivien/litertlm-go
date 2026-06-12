package litertlm

// The backend seam abstracts the inference engine behind Client and Chat so
// an alternative engine (litert-go's pure-Go `lm`) can serve the same API.
// The C++ LiteRT-LM engine is the default implementation (backend_cpp.go);
// engine selection happens at Client construction. The boundary speaks
// Go-native types only — C-handle construction and lifetime (SessionConfig,
// ConversationConfig, OptionalArgs, Responses, BenchmarkInfo) belong inside
// the C++ implementation.
//
// Chat's tool-dispatch and streaming machinery is already
// backend-parameterized through chatTransport (chat.go); ChatTransport
// extends that send surface with lifecycle and introspection.

// SessionSetup carries the Go-native inputs of a one-shot generation
// session: the per-call output cap and the resolved sampler.
type SessionSetup struct {
	MaxOutputTokens int
	Sampler         *SamplerParams
}

// ConversationSetup carries the Go-native inputs of a multi-turn
// conversation: the preface messages and knobs resolved before any engine
// work happens. The zero value is a bare conversation (the multimodal
// Generate path).
type ConversationSetup struct {
	SystemMessageJSON   string
	ToolsJSON           string
	MessagesJSON        string
	ConstrainedDecoding bool
	ExtraContextJSON    string
	// FilterChannelContentFromKVCache excludes reasoning-channel content
	// from the conversation's KV cache; nil keeps the engine default.
	FilterChannelContentFromKVCache *bool
	Sampler                         *SamplerParams
	MaxOutputTokens                 int
}

// RuntimeArgs carries the per-call arguments of one conversation send.
// The zero value means engine defaults.
type RuntimeArgs struct {
	VisualTokenBudget int // 0 = engine default
}

// Candidate is one generation result: the decoded text plus the optional
// scoring fields engines may provide (nil/absent when unsupported).
type Candidate struct {
	Text        string
	Score       *float32
	TokenLength *int
	TokenScores []float32
}

// Backend is the engine seam: everything Client needs from an inference
// engine. Implementations own their native resources; Close releases them.
type Backend interface {
	// Tokenize converts text to model token ids.
	Tokenize(text string) ([]int32, error)

	// NewSessionBackend opens a one-shot generation session (the
	// Generate / GenerateStream / GenerateResponse path).
	NewSessionBackend(s SessionSetup) (SessionBackend, error)

	// NewChatTransport opens a multi-turn conversation transport (the
	// NewChat path and the multimodal Generate path).
	NewChatTransport(s ConversationSetup) (ChatTransport, error)

	// Close releases the engine.
	Close()
}

// SessionBackend is one single-use generation session, owned by the caller.
type SessionBackend interface {
	// Generate runs one generation over parts and returns the candidates.
	Generate(parts []Part) ([]Candidate, error)
	// GenerateStreamCh streams chunks; the channel closes when decoding
	// completes, errors, or Cancel interrupts it.
	GenerateStreamCh(parts []Part) <-chan StreamChunk
	// ScoreTexts scores target strings against the session's context.
	ScoreTexts(targets []string, storeTokenLengths bool) ([]Candidate, error)
	// Benchmark snapshots engine-side timing; nil when benchmarking is
	// disabled or unsupported.
	Benchmark() *Benchmark
	// Cancel interrupts an in-flight generation; safe from another
	// goroutine (the ctx-cancellation hook).
	Cancel()
	Close()
}

// ChatTransport is the conversation seam Chat drives: the chatTransport
// send surface (with Go-native RuntimeArgs) plus lifecycle, cloning, and
// introspection.
type ChatTransport interface {
	// SendMessage sends one message envelope and returns the assistant's
	// raw reply envelope.
	SendMessage(messageJSON, extraContext string, args RuntimeArgs) (string, error)
	// SendMessageStreamCh is the streaming sibling; the channel closes
	// when the turn completes, errors, or Cancel interrupts it.
	SendMessageStreamCh(messageJSON, extraContext string, args RuntimeArgs) <-chan StreamChunk
	// Render returns the templated prompt for a message without running
	// inference.
	Render(messageJSON string) (string, error)
	// TokenCount reports the tokens accumulated in the conversation.
	TokenCount() (int, error)
	// Clone duplicates the conversation including its accumulated state.
	Clone() (ChatTransport, error)
	// Benchmark snapshots engine-side timing; nil when benchmarking is
	// disabled or unsupported.
	Benchmark() *Benchmark
	// Cancel interrupts an in-flight send; safe from another goroutine.
	Cancel()
	Close()
}
