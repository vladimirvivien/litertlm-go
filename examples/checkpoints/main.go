// checkpoints demonstrates low-level KV-cache branching using
// Session.RunPrefill, Session.SaveCheckpoint, Session.RunDecodeAsyncCh,
// Session.RewindToCheckpoint, and Session.Clone.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	model := flag.String("model", "", "path to .litertlm model file (falls back to LITERTLM_MODEL env)")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set (e.g. litert-community/gemma3-1b-it-int4)")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.19.1)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	prefix := flag.String("prefix", "In a small seaside village,", "shared prefix prefilled once before branching")
	maxTokens := flag.Int("max-output", 32, "max output tokens per branch")
	flag.Parse()

	ctx := context.Background()
	resolvedLib := *libPath
	if *getLib != "" {
		staged, err := litertlm.FetchLib(ctx, "", "", *getLib)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fetch library: %v\n", err)
			os.Exit(1)
		}
		resolvedLib = staged
	}

	resolvedModel := *model
	if resolvedModel == "" {
		resolvedModel = os.Getenv("LITERTLM_MODEL")
	}
	if *getModel != "" {
		staged, err := litertlm.FetchModel(ctx, *getModel)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fetch model: %v\n", err)
			os.Exit(1)
		}
		resolvedModel = staged
	}
	if resolvedModel == "" {
		fmt.Fprintln(os.Stderr, "--model or --get-model (or LITERTLM_MODEL env) is required")
		os.Exit(2)
	}

	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	client, err := litertlm.New(ctx,
		litertlm.WithLib(resolvedLib),
		litertlm.WithModel(resolvedModel),
		litertlm.WithBackend(*backend),
		litertlm.WithMaxTokens(1024),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new client: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = client.Close() }()

	sessCfg, err := litertlm.NewSessionConfig()
	if err != nil {
		fmt.Fprintf(os.Stderr, "new session config: %v\n", err)
		os.Exit(1)
	}
	defer sessCfg.Delete()
	if err = sessCfg.SetMaxOutputTokens(*maxTokens); err != nil {
		fmt.Fprintf(os.Stderr, "set max output tokens: %v\n", err)
		os.Exit(1)
	}

	session, err := client.Engine().NewSession(sessCfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new session: %v\n", err)
		os.Exit(1)
	}
	defer session.Delete()

	sharedIn, err := litertlm.NewTextInputString(*prefix)
	if err != nil {
		fmt.Fprintf(os.Stderr, "text input: %v\n", err)
		os.Exit(1)
	}
	defer sharedIn.Delete()

	if err = session.RunPrefill([]litertlm.InputData{sharedIn}); err != nil {
		fmt.Fprintf(os.Stderr, "shared prefill: %v\n", err)
		os.Exit(1)
	}
	if err = session.SaveCheckpoint("after-prefix"); err != nil {
		fmt.Fprintf(os.Stderr, "save checkpoint: %v\n", err)
		os.Exit(1)
	}

	cloned, err := session.Clone()
	if err != nil {
		fmt.Fprintf(os.Stderr, "clone session: %v\n", err)
		os.Exit(1)
	}
	defer cloned.Delete()

	if err = decodeBranch("Branch A (async decode)", session, " an old lighthouse keeper"); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}

	if err = session.RewindToCheckpoint("after-prefix"); err != nil {
		fmt.Fprintf(os.Stderr, "rewind to checkpoint: %v\n", err)
		os.Exit(1)
	}
	if err = decodeBranch("Branch B (after RewindToCheckpoint)", session, " a young boatbuilder"); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}

	if err = decodeBranch("Branch C (independent Session.Clone)", cloned, " a flock of silver gulls"); err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}

func decodeBranch(label string, s litertlm.Session, continuation string) error {
	in, err := litertlm.NewTextInputString(continuation)
	if err != nil {
		return fmt.Errorf("%s input: %w", label, err)
	}
	defer in.Delete()

	if err := s.RunPrefill([]litertlm.InputData{in}); err != nil {
		return fmt.Errorf("%s prefill: %w", label, err)
	}

	fmt.Printf("=== %s ===\n", label)
	for sc := range s.RunDecodeAsyncCh() {
		if sc.Err != nil {
			return fmt.Errorf("%s async decode: %w", label, sc.Err)
		}
		fmt.Print(sc.Text)
	}
	fmt.Println()
	return nil
}
