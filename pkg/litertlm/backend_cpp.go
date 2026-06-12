package litertlm

import "fmt"

// The C++ LiteRT-LM implementation of the backend seam. Each type wraps the
// corresponding C handle and keeps all handle construction and lifetime
// (SessionConfig, ConversationConfig, OptionalArgs, Responses,
// BenchmarkInfo) on this side of the boundary.

// cppBackend implements Backend over a constructed Engine. The Client owns
// engine construction (it also exposes the raw handle via Client.Engine);
// cppBackend owns nothing until Close, which releases the engine.
type cppBackend struct {
	engine           Engine
	benchmarkEnabled bool
}

func newCppBackend(engine Engine, benchmarkEnabled bool) *cppBackend {
	return &cppBackend{engine: engine, benchmarkEnabled: benchmarkEnabled}
}

func (b *cppBackend) Tokenize(text string) ([]int32, error) {
	return b.engine.Tokenize(text)
}

// sessionConfigFrom materializes a SessionConfig C handle from the Go-native
// setup, or the zero handle when nothing is set (engine defaults).
func sessionConfigFrom(maxOutputTokens int, sampler *SamplerParams) (SessionConfig, error) {
	if maxOutputTokens <= 0 && sampler == nil {
		return 0, nil
	}
	sessCfg, err := NewSessionConfig()
	if err != nil {
		return 0, err
	}
	if maxOutputTokens > 0 {
		sessCfg.SetMaxOutputTokens(maxOutputTokens)
	}
	if sampler != nil {
		sessCfg.SetSamplerParams(*sampler)
	}
	return sessCfg, nil
}

func (b *cppBackend) NewSessionBackend(s SessionSetup) (SessionBackend, error) {
	sessCfg, err := sessionConfigFrom(s.MaxOutputTokens, s.Sampler)
	if err != nil {
		return nil, err
	}
	sess, err := b.engine.NewSession(sessCfg)
	// SessionConfig fields are copied by NewSession; safe to delete now.
	sessCfg.Delete()
	if err != nil {
		return nil, err
	}
	return &cppSession{sess: sess, benchmarkEnabled: b.benchmarkEnabled}, nil
}

func (b *cppBackend) NewChatTransport(s ConversationSetup) (ChatTransport, error) {
	sessCfg, err := sessionConfigFrom(s.MaxOutputTokens, s.Sampler)
	if err != nil {
		return nil, err
	}
	convCfg, err := NewConversationConfig(b.engine, sessCfg,
		s.SystemMessageJSON, s.ToolsJSON, s.MessagesJSON, s.ConstrainedDecoding)
	// ConversationConfig copies sessCfg's relevant fields; safe to delete now.
	sessCfg.Delete()
	if err != nil {
		return nil, err
	}
	if err := convCfg.SetExtraContext(s.ExtraContextJSON); err != nil {
		convCfg.Delete()
		return nil, err
	}
	if s.FilterChannelContentFromKVCache != nil {
		convCfg.SetFilterChannelContentFromKVCache(*s.FilterChannelContentFromKVCache)
	}
	conv, err := b.engine.NewConversation(convCfg)
	if err != nil {
		convCfg.Delete()
		return nil, err
	}
	return &cppTransport{conv: conv, cfg: convCfg, benchmarkEnabled: b.benchmarkEnabled}, nil
}

func (b *cppBackend) Close() {
	b.engine.Delete()
	b.engine = 0
}

// cppSession implements SessionBackend over a Session handle.
type cppSession struct {
	sess             Session
	benchmarkEnabled bool
}

func (s *cppSession) Generate(parts []Part) ([]Candidate, error) {
	resp, err := s.sess.GenerateContent(partsToInputs(parts))
	if err != nil {
		return nil, err
	}
	defer resp.Delete()
	return harvestCandidates(resp), nil
}

func (s *cppSession) GenerateStreamCh(parts []Part) <-chan StreamChunk {
	return s.sess.GenerateContentStreamCh(partsToInputs(parts))
}

func (s *cppSession) ScoreTexts(targets []string, storeTokenLengths bool) ([]Candidate, error) {
	resp, err := s.sess.ScoreTexts(targets, storeTokenLengths)
	if err != nil {
		return nil, err
	}
	defer resp.Delete()
	return harvestCandidates(resp), nil
}

func (s *cppSession) Benchmark() *Benchmark {
	if !s.benchmarkEnabled {
		return nil
	}
	bi, err := s.sess.BenchmarkInfo()
	if err != nil {
		return nil
	}
	defer bi.Delete()
	return snapshotBenchmark(bi)
}

func (s *cppSession) Cancel() { s.sess.Cancel() }

func (s *cppSession) Close() {
	s.sess.Delete()
	s.sess = 0
}

// cppTransport implements ChatTransport over a Conversation handle and its
// owning ConversationConfig (which must outlive the conversation).
type cppTransport struct {
	conv             Conversation
	cfg              ConversationConfig
	benchmarkEnabled bool
}

// optionalArgsFrom materializes the per-call OptionalArgs handle, or the
// zero handle for engine defaults.
func optionalArgsFrom(args RuntimeArgs) (OptionalArgs, error) {
	if args.VisualTokenBudget <= 0 {
		return 0, nil
	}
	o, err := NewOptionalArgs()
	if err != nil {
		return 0, err
	}
	o.SetVisualTokenBudget(args.VisualTokenBudget)
	return o, nil
}

func (t *cppTransport) SendMessage(messageJSON, extraContext string, args RuntimeArgs) (string, error) {
	opt, err := optionalArgsFrom(args)
	if err != nil {
		return "", err
	}
	defer opt.Delete()
	return t.conv.SendMessage(messageJSON, extraContext, opt)
}

func (t *cppTransport) SendMessageStreamCh(messageJSON, extraContext string, args RuntimeArgs) <-chan StreamChunk {
	opt, err := optionalArgsFrom(args)
	if err != nil {
		out := make(chan StreamChunk, 1)
		out <- StreamChunk{Err: err, Final: true}
		close(out)
		return out
	}
	in := t.conv.SendMessageStreamCh(messageJSON, extraContext, opt)
	// The args handle must outlive the stream; release it when the C side
	// closes the channel.
	out := make(chan StreamChunk)
	go func() {
		defer close(out)
		defer opt.Delete()
		for sc := range in {
			out <- sc
		}
	}()
	return out
}

func (t *cppTransport) Render(messageJSON string) (string, error) {
	return t.conv.RenderMessage(messageJSON)
}

func (t *cppTransport) TokenCount() (int, error) {
	return t.conv.TokenCount()
}

func (t *cppTransport) Clone() (ChatTransport, error) {
	cloned, err := t.conv.Clone()
	if err != nil {
		return nil, fmt.Errorf("litertlm: clone conversation: %w", err)
	}
	// The clone shares the original's ConversationConfig, whose lifetime
	// stays with the original transport.
	return &cppTransport{conv: cloned, benchmarkEnabled: t.benchmarkEnabled}, nil
}

func (t *cppTransport) Benchmark() *Benchmark {
	if !t.benchmarkEnabled {
		return nil
	}
	bi, err := t.conv.BenchmarkInfo()
	if err != nil {
		return nil
	}
	defer bi.Delete()
	return snapshotBenchmark(bi)
}

func (t *cppTransport) Cancel() { t.conv.Cancel() }

func (t *cppTransport) Close() {
	t.conv.Delete()
	t.cfg.Delete()
	t.conv = 0
	t.cfg = 0
}

// harvestCandidates copies a Responses handle's candidates into Go-native
// Candidate values so the handle can be released immediately.
func harvestCandidates(resp Responses) []Candidate {
	n := resp.NumCandidates()
	out := make([]Candidate, 0, n)
	for i := 0; i < n; i++ {
		c := Candidate{Text: resp.Text(i)}
		if score, ok := resp.Score(i); ok {
			c.Score = &score
		}
		if tl, ok := resp.TokenLength(i); ok {
			c.TokenLength = &tl
		}
		if ts, ok := resp.TokenScores(i); ok {
			c.TokenScores = ts
		}
		out = append(out, c)
	}
	return out
}
