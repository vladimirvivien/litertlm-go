// Command engdiff is the cross-engine differential harness: it runs the same
// prompt, greedy, on the C++ LiteRT-LM engine and on the pure-Go engine
// (litert-go), and diffs the replies. Greedy CPU decoding is deterministic
// in both engines, so byte equality is the expected result; any divergence
// localizes a templating, tokenization, or numeric difference between the
// engines.
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
//	engdiff -models <dir>                 # diff every .litertlm in dir
//	engdiff -models a.litertlm,b.litertlm # diff a list
//
// Library directories come from -litertlm-lib / -litert-lib or the
// LITERTLM_LIB / LITERT_LIB environment variables.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	golm "github.com/vladimirvivien/litert-go/lm"
	"github.com/vladimirvivien/litertlm-go/backend/litertgo"
	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

const (
	replyBegin = "<<ENGDIFF-REPLY>>"
	replyEnd   = "<<ENGDIFF-END>>"
)

func main() {
	models := flag.String("models", "", "directory of .litertlm files, or comma-separated paths")
	prompt := flag.String("prompt", "Name one primary color.", "user prompt (sent as one chat turn)")
	cppLib := flag.String("litertlm-lib", os.Getenv("LITERTLM_LIB"), "LiteRT-LM shared-library directory")
	goLib := flag.String("litert-lib", os.Getenv("LITERT_LIB"), "litert-go runtime-library directory")
	run := flag.String("run", "", "worker mode: cpp or go (single model, prints reply)")
	model := flag.String("model", "", "worker mode: model path")
	flag.Parse()

	if *run != "" {
		if err := worker(*run, *model, *prompt, *cppLib, *goLib); err != nil {
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

// capture re-executes this binary in worker mode and extracts the reply
// between the markers, keeping engine logs out of the comparison.
func capture(engine, model, prompt string, cppLib, goLib string) (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	cmd := exec.Command(exe, "-run", engine, "-model", model, "-prompt", prompt,
		"-litertlm-lib", cppLib, "-litert-lib", goLib)
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("%s worker: %w", engine, err)
	}
	s := string(out)
	i := strings.Index(s, replyBegin)
	j := strings.LastIndex(s, replyEnd)
	if i < 0 || j < 0 || j < i {
		return "", fmt.Errorf("%s worker: no reply markers in output", engine)
	}
	return s[i+len(replyBegin) : j], nil
}

func worker(engine, model, prompt string, cppLib, goLib string) error {
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

	client, err := litertlm.New(ctx, opts...)
	if err != nil {
		return err
	}
	defer client.Close()
	chat, err := client.NewChat(ctx)
	if err != nil {
		return err
	}
	r, err := chat.Send(ctx, prompt)
	if err != nil {
		return err
	}
	fmt.Print(replyBegin, r.Text(), replyEnd)
	return nil
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
