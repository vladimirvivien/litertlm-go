// Command engdiff is the cross-engine differential harness: it runs the same
// prompts, greedy, on the C++ LiteRT-LM engine and on the pure-Go engine
// (litert-go), and compares the replies. Greedy CPU decoding is deterministic
// in both engines.
//
// Both workers drive the same litertlm-go Client / Chat code path; they
// differ only in backend construction (the C++ engine by default, the
// litert-go engine via WithEngineBackend). A divergence therefore
// localizes to the engine, not to API-layer differences.
//
// Each engine runs in its own subprocess (the harness re-executes itself
// with -run): the two stacks load different LiteRT runtime libraries that
// must not share a process.
//
// Modes:
//
//	engdiff -models <dir>                  # Tier 1: byte-diff one chat turn per model
//	engdiff -models <dir> -eval set.json   # Tier 2: task-accuracy eval, cross-engine delta gate
//
// The eval mode also reports Tier 3 performance per engine (init seconds,
// end-to-end reply tokens/sec) and, with -perf-baseline, gates on >10%
// regression against a recorded run (-perf-out writes one).
//
// Library directories come from -litertlm-lib / -litert-lib or the
// LITERTLM_LIB / LITERT_LIB environment variables.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	golm "github.com/vladimirvivien/litert-go/lm"
	"github.com/vladimirvivien/litertlm-go/backend/litertgo"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

const (
	replyBegin = "<<ENGDIFF-REPLY>>"
	replyEnd   = "<<ENGDIFF-END>>"
	itemBegin  = "<<ENGDIFF-ITEM " // followed by "<index>>>"
	perfBegin  = "<<ENGDIFF-PERF>>"
)

func main() {
	models := flag.String("models", "", "directory of .litertlm files, or comma-separated paths")
	prompt := flag.String("prompt", "Name one primary color.", "user prompt (sent as one chat turn)")
	evalSet := flag.String("eval", "", "eval-set JSON file: run task-accuracy eval instead of single-prompt diff")
	delta := flag.Float64("delta", 5.0, "eval gate: max cross-engine accuracy delta in percentage points")
	perfOut := flag.String("perf-out", "", "append per-engine perf records (JSONL) to this file")
	perfBaseline := flag.String("perf-baseline", "", "perf JSONL to gate against (>10% init or tok/s regression fails)")
	cppLib := flag.String("litertlm-lib", os.Getenv("LITERTLM_LIB"), "LiteRT-LM shared-library directory")
	goLib := flag.String("litert-lib", os.Getenv("LITERT_LIB"), "litert-go runtime-library directory")
	run := flag.String("run", "", "worker mode: cpp or go (single model)")
	model := flag.String("model", "", "worker mode: model path")
	prompts := flag.String("prompts", "", "worker mode: JSON file holding a prompt array (batch)")
	flag.Parse()

	if *run != "" {
		if err := worker(*run, *model, *prompt, *prompts, *cppLib, *goLib); err != nil {
			fmt.Fprintln(os.Stderr, "engdiff worker:", err)
			os.Exit(1)
		}
		return
	}
	if *models == "" {
		flag.Usage()
		os.Exit(2)
	}
	files, err := modelList(*models)
	if err != nil {
		fmt.Fprintln(os.Stderr, "engdiff:", err)
		os.Exit(1)
	}

	if *evalSet != "" {
		ok := runEval(files, *evalSet, *delta, *perfOut, *perfBaseline, *cppLib, *goLib)
		if !ok {
			os.Exit(1)
		}
		return
	}

	pass, fail := 0, 0
	for _, m := range files {
		cpp, cppErr := capture("cpp", m, *prompt, *cppLib, *goLib)
		gor, goErr := capture("go", m, *prompt, *cppLib, *goLib)
		name := filepath.Base(m)
		switch {
		case cppErr != nil || goErr != nil:
			fail++
			fmt.Printf("ERROR %-45s cpp=%v go=%v\n", name, cppErr, goErr)
		case cpp == gor:
			pass++
			fmt.Printf("MATCH %-45s %q\n", name, clip(cpp, 60))
		default:
			fail++
			fmt.Printf("DIFF  %-45s\n  cpp: %q\n  go:  %q\n  first divergence at byte %d\n",
				name, clip(cpp, 120), clip(gor, 120), divergeAt(cpp, gor))
		}
	}
	fmt.Printf("\n%d match, %d differ/error of %d models\n", pass, fail, pass+fail)
	if fail > 0 {
		os.Exit(1)
	}
}

// capture re-executes this binary in single-prompt worker mode and extracts
// the reply between the markers, keeping engine logs out of the comparison.
func capture(engine, model, prompt, cppLib, goLib string) (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	cmd := exec.Command(exe, "-run", engine, "-model", model, "-prompt", prompt,
		"-litertlm-lib", cppLib, "-litert-lib", goLib)
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("%s worker: %w%s", engine, err, stderrTail(err))
	}
	s := string(out)
	i := strings.Index(s, replyBegin)
	j := strings.LastIndex(s, replyEnd)
	if i < 0 || j < 0 || j < i {
		return "", fmt.Errorf("%s worker: no reply markers in output", engine)
	}
	return s[i+len(replyBegin) : j], nil
}

// perfRecord is one engine's Tier 3 measurement over a batch run.
type perfRecord struct {
	Model       string  `json:"model"`
	Engine      string  `json:"engine"`
	InitSecs    float64 `json:"init_secs"`
	ReplyTokens int     `json:"reply_tokens"`
	ReplySecs   float64 `json:"reply_secs"`
	TokPerSec   float64 `json:"tok_per_sec"`
}

// captureBatch runs one engine worker over a prompt list (one model load,
// one fresh chat per prompt) and returns the replies plus the perf record.
func captureBatch(engine, model string, promptList []string, cppLib, goLib string) ([]string, perfRecord, error) {
	exe, err := os.Executable()
	if err != nil {
		return nil, perfRecord{}, err
	}
	f, err := os.CreateTemp("", "engdiff-prompts-*.json")
	if err != nil {
		return nil, perfRecord{}, err
	}
	defer os.Remove(f.Name())
	if err := json.NewEncoder(f).Encode(promptList); err != nil {
		f.Close()
		return nil, perfRecord{}, err
	}
	f.Close()

	cmd := exec.Command(exe, "-run", engine, "-model", model, "-prompts", f.Name(),
		"-litertlm-lib", cppLib, "-litert-lib", goLib)
	out, err := cmd.Output()
	if err != nil {
		return nil, perfRecord{}, fmt.Errorf("%s worker: %w%s", engine, err, stderrTail(err))
	}

	replies := make([]string, len(promptList))
	s := string(out)
	for i := range promptList {
		tag := fmt.Sprintf("%s%d>>", itemBegin, i)
		a := strings.Index(s, tag)
		if a < 0 {
			return nil, perfRecord{}, fmt.Errorf("%s worker: item %d missing", engine, i)
		}
		rest := s[a+len(tag):]
		b := strings.Index(rest, replyEnd)
		if b < 0 {
			return nil, perfRecord{}, fmt.Errorf("%s worker: item %d unterminated", engine, i)
		}
		replies[i] = rest[:b]
	}

	var perf perfRecord
	if a := strings.Index(s, perfBegin); a >= 0 {
		rest := s[a+len(perfBegin):]
		if b := strings.Index(rest, replyEnd); b >= 0 {
			if err := json.Unmarshal([]byte(rest[:b]), &perf); err != nil {
				return nil, perfRecord{}, fmt.Errorf("%s worker: bad perf record: %w", engine, err)
			}
		}
	}
	perf.Model = filepath.Base(model)
	perf.Engine = engine
	return replies, perf, nil
}

// worker runs inside the engine subprocess. With -prompts it answers a
// batch (one client, one fresh chat per prompt) and emits per-item
// markers plus a perf record; otherwise it answers the single -prompt.
func worker(engine, model, prompt, promptsFile, cppLib, goLib string) error {
	ctx := context.Background()

	var opts []litertlm.Option
	switch engine {
	case "cpp":
		opts = []litertlm.Option{
			litertlm.WithLib(cppLib), litertlm.WithModel(model),
			litertlm.WithBackend("cpu"),
		}
	case "go":
		b, err := litertgo.Open(ctx, model, golm.WithLibDir(goLib))
		if err != nil {
			return err
		}
		opts = []litertlm.Option{litertlm.WithEngineBackend(b)}
	default:
		return fmt.Errorf("unknown engine %q", engine)
	}

	initStart := time.Now()
	client, err := litertlm.New(ctx, opts...)
	if err != nil {
		return err
	}
	defer client.Close()
	initSecs := time.Since(initStart).Seconds()

	oneTurn := func(p string) (string, error) {
		chat, err := client.NewChat(ctx)
		if err != nil {
			return "", err
		}
		defer chat.Close()
		r, err := chat.Send(ctx, p)
		if err != nil {
			return "", err
		}
		return r.Text(), nil
	}

	if promptsFile == "" {
		reply, err := oneTurn(prompt)
		if err != nil {
			return err
		}
		fmt.Print(replyBegin, reply, replyEnd)
		return nil
	}

	data, err := os.ReadFile(promptsFile)
	if err != nil {
		return err
	}
	var promptList []string
	if err := json.Unmarshal(data, &promptList); err != nil {
		return fmt.Errorf("parse prompts: %w", err)
	}

	perf := perfRecord{InitSecs: initSecs}
	for i, p := range promptList {
		start := time.Now()
		reply, err := oneTurn(p)
		if err != nil {
			return fmt.Errorf("item %d: %w", i, err)
		}
		perf.ReplySecs += time.Since(start).Seconds()
		if toks, err := client.Tokenize(reply); err == nil {
			perf.ReplyTokens += len(toks)
		}
		fmt.Printf("%s%d>>%s%s\n", itemBegin, i, reply, replyEnd)
	}
	if perf.ReplySecs > 0 {
		perf.TokPerSec = float64(perf.ReplyTokens) / perf.ReplySecs
	}
	b, err := json.Marshal(perf)
	if err != nil {
		return err
	}
	fmt.Print(perfBegin, string(b), replyEnd)
	return nil
}

// stderrTail extracts the last lines of a failed worker's stderr so
// engine errors survive the subprocess boundary.
func stderrTail(err error) string {
	var ee *exec.ExitError
	if !errors.As(err, &ee) || len(ee.Stderr) == 0 {
		return ""
	}
	lines := strings.Split(strings.TrimSpace(string(ee.Stderr)), "\n")
	if len(lines) > 4 {
		lines = lines[len(lines)-4:]
	}
	return "\n    " + strings.Join(lines, "\n    ")
}

func modelList(arg string) ([]string, error) {
	if fi, err := os.Stat(arg); err == nil && fi.IsDir() {
		files, err := filepath.Glob(filepath.Join(arg, "*.litertlm"))
		if err != nil || len(files) == 0 {
			return nil, fmt.Errorf("no .litertlm files under %s", arg)
		}
		return files, nil
	}
	return strings.Split(arg, ","), nil
}

func clip(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

func divergeAt(a, b string) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}
