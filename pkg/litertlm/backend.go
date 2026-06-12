package litertlm

// The backend seam abstracts the inference engine behind Client and Chat so
// an alternative engine (litert-go's pure-Go `lm`) can serve the same API.
// The C++ LiteRT-LM engine is the default implementation; engine selection
// happens at Client construction. The boundary speaks Go-native types only —
// C-handle construction (SessionConfig, ConversationConfig, OptionalArgs)
// belongs inside the C++ implementation.
//
// Chat's tool-dispatch and streaming machinery is already
// backend-parameterized through chatTransport (chat.go); Backend adds the
// engine- and session-level operations plus transport construction.

// SessionSetup carries the Go-native inputs of a one-shot generation
// session: the per-call output cap and the resolved sampler (per-call over
// client default).
type SessionSetup struct {
	MaxOutputTokens int
	Sampler         *SamplerParams
}

// ConversationSetup carries the Go-native inputs of a multi-turn
// conversation: the preface messages and knobs Chat construction resolves
// before any engine work happens.
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
}

// Candidate is one generation result: the decoded text plus the optional
// scoring fields engines may provide (nil/zero when unsupported).
type Candidate struct {
	Text        string
	Score       *float32
	TokenLength int
	TokenScores []float32
}

// Backend is the engine seam: everything Client needs from an inference
// engine. Implementations own their native resources; Close releases them.
type Backend interface {
	// Tokenize converts text to model token ids (used by Client.Tokenize
	// and TokenLength).
	Tokenize(text string) ([]int32, error)

	// NewSessionBackend opens a one-shot generation session (the
	// Generate / GenerateStream / GenerateResponse path).
	NewSessionBackend(s SessionSetup) (SessionBackend, error)

	// NewChatTransport opens a multi-turn conversation transport (the
	// NewChat path). The returned transport feeds Chat's
	// dispatch/streaming machinery (chatTransport) and adds lifecycle.
	NewChatTransport(s ConversationSetup) (ChatTransport, error)

	// Close releases the engine. Idempotence is the caller's concern.
	Close()
}

// SessionBackend is one generation session. Sessions are single-use
// resources owned by the caller of NewSessionBackend.
type SessionBackend interface {
	// Generate runs one generation over parts and returns the candidates.
	Generate(parts []Part) ([]Candidate, error)
	// GenerateStream streams chunks through cb and returns when decoding
	// completes or Cancel interrupts it.
	GenerateStream(parts []Part, cb func(StreamChunk)) error
	// ScoreTexts scores target strings against the session's prefilled
	// context (the GenerateResponse scoring path).
	ScoreTexts(targets []string, storeTokenLengths bool) ([]Candidate, error)
	// BenchmarkInfo reports engine-side timing when benchmarking is
	// enabled.
	BenchmarkInfo() (BenchmarkInfo, error)
	// Cancel interrupts an in-flight Generate/GenerateStream; safe from
	// another goroutine (the ctx-cancellation hook).
	Cancel()
	Close()
}

// ChatTransport is the conversation seam Chat drives: chatTransport's
// send surface plus lifecycle and prompt introspection.
type ChatTransport interface {
	chatTransport // SendMessage / SendMessageStreamCh (chat.go)

	// Render returns the templated prompt for a message without running
	// inference (Chat.RenderMessage).
	Render(messageJSON string) (string, error)
	// Cancel interrupts an in-flight send; safe from another goroutine.
	Cancel()
	Close()
}
