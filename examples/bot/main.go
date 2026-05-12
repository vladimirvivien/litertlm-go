// bot is a tiny CLI chat assistant. Each run is a REPL session that
// streams replies token-by-token; conversation memory persists across
// runs in MEM.log, and the bot uses its own inference to compact that
// memory when it would overflow the engine's context budget.
//
// See README.md in this directory for prerequisites and usage.
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"strings"
	"syscall"
	"time"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

const (
	defaultSystemPrompt = "You are a helpful and delightful assistant happy to help the user."

	compactSystemPrompt = `You are a memory compactor. Rewrite the following conversation transcript ` +
		`into a brief running summary. Preserve facts the user shared about themselves, their ` +
		`stated preferences, decisions made, and the current topic. Drop pleasantries, repeated ` +
		`information, and one-off questions whose answers no longer matter. Output only the ` +
		`summary text — no preamble, no commentary.`
)

func main() {
	model := flag.String("model", "", "path to .litertlm chat-tuned model file")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding the LiteRT-LM shared libraries")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	systemFlag := flag.String("system", "", "inline system prompt (overrides -system-file and SYSTEM.md)")
	systemFile := flag.String("system-file", "", "path to system prompt file (overrides SYSTEM.md)")
	memPath := flag.String("mem", "", "path to a memory file to persist conversation across runs (e.g. MEM.log). Empty = ephemeral chat (default)")
	maxTokens := flag.Int("max", 4096, "engine max tokens (prompt + output); raise toward the model's context ceiling (32K for Gemma 4) for longer chats — KV cache grows linearly")
	prompt := flag.String("prompt", "", "if set, send this single message and exit (one-shot mode)")
	speculative := flag.Bool("speculative", false, "enable multi-token-prediction speculative decoding")
	reset := flag.Bool("reset", false, "truncate the memory file before starting (requires -mem)")
	compactNow := flag.Bool("compact-now", false, "force a compaction at startup (requires -mem)")
	compactAt := flag.Float64("compact-at", 0.80, "compact when projected tokens exceed this fraction of -max")
	replyReserve := flag.Int("reply-reserve", 1024, "tokens reserved for the model's reply when budgeting")
	temperature := flag.Float64("temperature", 0.7, "sampling temperature; 0 = greedy (prone to repetition loops on small models)")
	topP := flag.Float64("top-p", 0.95, "nucleus sampling cutoff (used when temperature > 0)")
	topK := flag.Int("top-k", 40, "top-k sampling cutoff (used when temperature > 0)")
	seed := flag.Int("seed", 0, "sampler RNG seed (0 = nondeterministic)")
	flag.Parse()

	if *model == "" {
		fmt.Fprintln(os.Stderr, "--model is required")
		os.Exit(2)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	systemPrompt, err := resolveSystemPrompt(*systemFlag, *systemFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "system prompt: %v\n", err)
		os.Exit(1)
	}

	if *reset && *memPath != "" {
		if err = os.WriteFile(*memPath, nil, 0o644); err != nil && !errors.Is(err, os.ErrNotExist) {
			fmt.Fprintf(os.Stderr, "reset memory: %v\n", err)
			os.Exit(1)
		}
	}

	opts := []litertlm.Option{
		litertlm.WithLib(*libPath),
		litertlm.WithModel(*model),
		litertlm.WithBackend(*backend),
		litertlm.WithMaxTokens(*maxTokens),
	}
	if *speculative {
		opts = append(opts, litertlm.WithSpeculativeDecodingEnabled(true))
	}
	sampler := litertlm.DefaultSamplerParams()
	if *temperature > 0 {
		sampler.Type = litertlm.SamplerTopP
		sampler.Temperature = float32(*temperature)
		sampler.TopP = float32(*topP)
		sampler.TopK = int32(*topK)
		sampler.Seed = int32(*seed)
	}
	opts = append(opts, litertlm.WithDefaultSampler(sampler))

	client, err := litertlm.New(ctx, opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = client.Close() }()

	bot := &bot{
		client:        client,
		modelName:     filepath.Base(*model),
		systemPrompt:  systemPrompt,
		memPath:       *memPath,
		maxTokens:     *maxTokens,
		compactAt:     *compactAt,
		replyReserve:  *replyReserve,
		compactPrompt: compactSystemPrompt,
	}

	if err := bot.loadMemory(); err != nil {
		fmt.Fprintf(os.Stderr, "load memory: %v\n", err)
		os.Exit(1)
	}

	if *compactNow && len(bot.turns) > 0 {
		if err := bot.compact(ctx); err != nil {
			fmt.Fprintf(os.Stderr, "compact: %v\n", err)
			os.Exit(1)
		}
	}

	if err := bot.warmup(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "warmup: %v\n", err)
		os.Exit(1)
	}

	if err := bot.openChat(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "new chat: %v\n", err)
		os.Exit(1)
	}
	defer bot.closeChat()

	bot.printBanner()

	if *prompt != "" {
		fmt.Println()
		fmt.Printf("〉 %s\n", *prompt)
		if err := bot.handle(ctx, *prompt); err != nil {
			fmt.Fprintf(os.Stderr, "send: %v\n", err)
			os.Exit(1)
		}
		return
	}

	if err := bot.repl(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "repl: %v\n", err)
		os.Exit(1)
	}
}

func isExitCommand(line string) bool {
	s := strings.ToLower(strings.TrimSpace(line))
	return s == "exit" || s == "/exit"
}

func resolveSystemPrompt(inline, file string) (string, error) {
	if inline != "" {
		return inline, nil
	}
	if file != "" {
		b, err := os.ReadFile(file)
		if err != nil {
			return "", fmt.Errorf("read %s: %w", file, err)
		}
		return strings.TrimSpace(string(b)), nil
	}
	if b, err := os.ReadFile("SYSTEM.md"); err == nil {
		return strings.TrimSpace(string(b)), nil
	}
	return defaultSystemPrompt, nil
}

// turn is one user/assistant exchange in the persisted transcript.
type turn struct {
	role string
	ts   time.Time
	body string
}

type bot struct {
	client       *litertlm.Client
	chat         *litertlm.Chat
	modelName    string
	systemPrompt string

	memPath string
	turns   []turn

	maxTokens     int
	compactAt     float64
	replyReserve  int
	compactPrompt string
}

func (b *bot) printBanner() {
	tokens, err := b.client.TokenLength(b.effectiveSystemPrompt())
	if err != nil {
		fmt.Fprintf(os.Stderr, "warn: token count: %v\n", err)
	}
	memDesc := "ephemeral"
	if b.memPath != "" {
		memDesc = fmt.Sprintf("%s (%d turns)", b.memPath, len(b.turns))
	}
	fmt.Fprintf(os.Stderr, "🤖 %s\n", b.modelName)
	fmt.Fprintf(os.Stderr, "💾 %s · 📝 %d system tokens\n", memDesc, tokens)
	fmt.Fprintf(os.Stderr, "📐 %d max · %d reply reserve\n", b.maxTokens, b.replyReserve)
}

func (b *bot) repl(ctx context.Context) error {
	fmt.Println()
	fmt.Println("🤖 How can I help you today?")

	// stdin reads block in a syscall that does not respect ctx, so the
	// read happens on a background goroutine and the main loop selects
	// on it alongside ctx.Done(). On exit the goroutine is leaked but
	// the process is on its way out anyway.
	lines := make(chan string)
	readErrs := make(chan error, 1)
	go func() {
		defer close(lines)
		r := bufio.NewReader(os.Stdin)
		for {
			line, err := r.ReadString('\n')
			if err != nil {
				if !errors.Is(err, io.EOF) {
					readErrs <- err
				}
				return
			}
			lines <- strings.TrimRight(line, "\r\n")
		}
	}()

	for {
		fmt.Println()
		fmt.Print("〉 ")
		select {
		case <-ctx.Done():
			fmt.Println()
			return nil
		case err := <-readErrs:
			return err
		case line, ok := <-lines:
			if !ok {
				fmt.Println()
				return nil
			}
			if line == "" {
				continue
			}
			if isExitCommand(line) {
				return nil
			}
			if err := b.handle(ctx, line); err != nil {
				if errors.Is(err, context.Canceled) {
					fmt.Println()
					return nil
				}
				return err
			}
		}
	}
}

func (b *bot) handle(ctx context.Context, msg string) error {
	if err := b.maybeCompact(ctx, msg); err != nil {
		return err
	}
	if err := b.ensureChat(ctx); err != nil {
		return fmt.Errorf("open chat after compaction: %w", err)
	}

	fmt.Println()
	fmt.Print("🤖 ")
	var reply strings.Builder
	for chunk, err := range b.chat.SendStream(ctx, msg) {
		if err != nil {
			fmt.Println()
			return fmt.Errorf("stream: %w", err)
		}
		fmt.Print(chunk.Text)
		reply.WriteString(chunk.Text)
		if chunk.Final {
			fmt.Println()
		}
	}

	if b.memPath != "" {
		now := time.Now().UTC()
		b.turns = append(b.turns,
			turn{role: "user", ts: now, body: msg},
			turn{role: "bot", ts: now, body: reply.String()},
		)
		if err := b.appendTurnsToFile(b.turns[len(b.turns)-2:]); err != nil {
			return fmt.Errorf("append memory: %w", err)
		}
	}
	return nil
}

func (b *bot) maybeCompact(ctx context.Context, nextUserMsg string) error {
	if b.memPath == "" || len(b.turns) == 0 {
		return nil
	}
	sysTokens, err := b.client.TokenLength(b.effectiveSystemPrompt())
	if err != nil {
		return fmt.Errorf("token-count system: %w", err)
	}
	userTokens, err := b.client.TokenLength(nextUserMsg)
	if err != nil {
		return fmt.Errorf("token-count user: %w", err)
	}
	projected := sysTokens + userTokens + b.replyReserve
	threshold := int(float64(b.maxTokens) * b.compactAt)
	if projected <= threshold {
		return nil
	}
	return b.compact(ctx)
}

func (b *bot) compact(ctx context.Context) error {
	transcript := renderTurns(b.turns)
	beforeTokens, _ := b.client.TokenLength(transcript)

	fmt.Fprintf(os.Stderr, "[compacting memory: %d turns, %d tokens", len(b.turns), beforeTokens)

	// The C engine permits one active session at a time, so the main
	// chat must be closed before opening the compaction chat. ensureChat
	// reopens it on demand the next time handle() runs.
	b.closeChat()

	tmp, err := b.client.NewChat(ctx, litertlm.WithSystemPrompt(b.compactPrompt))
	if err != nil {
		fmt.Fprintln(os.Stderr, "]")
		return fmt.Errorf("compact: new chat: %w", err)
	}
	defer func() { _ = tmp.Close() }()

	reply, err := tmp.Send(ctx, "Compact this transcript:\n\n"+transcript)
	if err != nil {
		fmt.Fprintln(os.Stderr, "]")
		return fmt.Errorf("compact: send: %w", err)
	}
	summary := strings.TrimSpace(reply.Text())

	now := time.Now().UTC()
	compacted := []turn{{role: "bot", ts: now, body: summary}}
	if err := b.writeMemoryFile(compacted); err != nil {
		fmt.Fprintln(os.Stderr, "]")
		return fmt.Errorf("compact: write: %w", err)
	}
	b.turns = compacted

	afterTokens, _ := b.client.TokenLength(summary)
	fmt.Fprintf(os.Stderr, " → %d tokens]\n", afterTokens)
	return nil
}

// warmup runs one tiny throwaway inference against a temp Chat so the
// engine pays its first-inference fixed cost (TFLite executor lazy
// init, allocator preallocation, mmap page-in) at startup instead of
// on the user's first prompt. With ephemeral memory the system prompt
// is small, so this is the dominant startup cost worth pre-paying.
func (b *bot) warmup(ctx context.Context) error {
	fmt.Fprint(os.Stderr, "Waking up ⏱️... ")
	start := time.Now()

	tmp, err := b.client.NewChat(ctx, litertlm.WithSystemPrompt("warmup"))
	if err != nil {
		fmt.Fprintln(os.Stderr, "skipped")
		return fmt.Errorf("new chat: %w", err)
	}
	defer func() { _ = tmp.Close() }()

	if _, err := tmp.Send(ctx, "hi"); err != nil {
		fmt.Fprintln(os.Stderr, "skipped")
		return fmt.Errorf("send: %w", err)
	}

	fmt.Fprintf(os.Stderr, "done in %s\n", time.Since(start).Round(time.Millisecond))
	return nil
}

func (b *bot) ensureChat(ctx context.Context) error {
	if b.chat != nil {
		return nil
	}
	return b.openChat(ctx)
}

func (b *bot) openChat(ctx context.Context) error {
	chat, err := b.client.NewChat(ctx, litertlm.WithSystemPrompt(b.effectiveSystemPrompt()))
	if err != nil {
		return err
	}
	b.chat = chat
	return nil
}

func (b *bot) closeChat() {
	if b.chat != nil {
		_ = b.chat.Close()
		b.chat = nil
	}
}

func (b *bot) effectiveSystemPrompt() string {
	if len(b.turns) == 0 {
		return b.systemPrompt
	}
	var sb strings.Builder
	sb.WriteString(b.systemPrompt)
	sb.WriteString("\n\n--- Prior conversation memory ---\n")
	sb.WriteString(renderTurns(b.turns))
	sb.WriteString("--- End memory ---\n")
	return sb.String()
}

// ---- MEM.log I/O -----------------------------------------------------------

const fence = "````"

var headerRe = regexp.MustCompile("^> (user|bot) (\\S+)`{4,}$")

func (b *bot) loadMemory() error {
	if b.memPath == "" {
		return nil
	}
	data, err := os.ReadFile(b.memPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	turns, err := parseMemory(string(data))
	if err != nil {
		return err
	}
	b.turns = turns
	return nil
}

func parseMemory(s string) ([]turn, error) {
	var turns []turn
	lines := strings.Split(s, "\n")
	i := 0
	for i < len(lines) {
		line := lines[i]
		if line == "" {
			i++
			continue
		}
		m := headerRe.FindStringSubmatch(line)
		if m == nil {
			return nil, fmt.Errorf("memory parse: line %d: not a turn header: %q", i+1, line)
		}
		role := m[1]
		ts, err := time.Parse(time.RFC3339, m[2])
		if err != nil {
			return nil, fmt.Errorf("memory parse: line %d: bad timestamp %q: %w", i+1, m[2], err)
		}
		i++
		var body strings.Builder
		closed := false
		for i < len(lines) {
			if isFenceLine(lines[i]) {
				closed = true
				i++
				break
			}
			if body.Len() > 0 {
				body.WriteByte('\n')
			}
			body.WriteString(lines[i])
			i++
		}
		if !closed {
			return nil, fmt.Errorf("memory parse: turn at line ending without closing fence")
		}
		turns = append(turns, turn{role: role, ts: ts, body: body.String()})
	}
	return turns, nil
}

func isFenceLine(line string) bool {
	if len(line) < 4 {
		return false
	}
	for _, r := range line {
		if r != '`' {
			return false
		}
	}
	return true
}

func (b *bot) appendTurnsToFile(turns []turn) error {
	f, err := os.OpenFile(b.memPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	if _, err := f.WriteString(renderTurns(turns)); err != nil {
		return err
	}
	return nil
}

func (b *bot) writeMemoryFile(turns []turn) error {
	return os.WriteFile(b.memPath, []byte(renderTurns(turns)), 0o644)
}

func renderTurns(turns []turn) string {
	var sb strings.Builder
	for _, t := range turns {
		fmt.Fprintf(&sb, "> %s %s%s\n", t.role, t.ts.Format(time.RFC3339), fence)
		sb.WriteString(t.body)
		if !strings.HasSuffix(t.body, "\n") {
			sb.WriteByte('\n')
		}
		sb.WriteString(fence)
		sb.WriteByte('\n')
	}
	return sb.String()
}
