// debugger demonstrates enabling the AI Edge Debugger via WithDebuggerEnabled
// (LiteRT-LM C API 1.1.0+), pairing it with a writable cache directory, and
// inspecting relative trace capture locations via Chat.DebugCaptureDir and
// Session.DebugCaptureDir.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func main() {
	model := flag.String("model", "", "path to a .litertlm model file (or set LITERTLM_MODEL)")
	getModel := flag.String("get-model", "", "download model from Hugging Face or URL if set")
	getLib := flag.String("get-lib", "", "download LiteRT-LM shared library version if set (e.g. v0.19.1)")
	libPath := flag.String("lib", os.Getenv("LITERTLM_LIB"), "directory holding LiteRT-LM shared libs (falls back to LITERTLM_LIB env)")
	backend := flag.String("backend", "cpu", "inference backend (cpu | gpu)")
	cacheDir := flag.String("cache-dir", filepath.Join(os.TempDir(), "litertlm-debug-cache"), "writable cache directory for debugger traces")
	enableDebugger := flag.Bool("enable-debugger", false, "enable AI Edge Debugger traces via WithDebuggerEnabled (requires C API 1.1.0+)")
	prompt := flag.String("prompt", "Write a three-word greeting.", "prompt to trace")
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
		fmt.Fprintln(os.Stderr, "-model or -get-model (or LITERTLM_MODEL env) is required")
		os.Exit(2)
	}

	if err := os.MkdirAll(*cacheDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "create cache dir: %v\n", err)
		os.Exit(1)
	}

	litertlm.SetMinLogLevel(litertlm.LogQuiet)
	opts := []litertlm.Option{
		litertlm.WithLib(resolvedLib),
		litertlm.WithModel(resolvedModel),
		litertlm.WithBackend(*backend),
		litertlm.WithCacheDir(*cacheDir),
	}
	if *enableDebugger {
		opts = append(opts, litertlm.WithDebuggerEnabled(true))
	}

	client, err := litertlm.New(ctx, opts...)
	if err != nil {
		if *enableDebugger && errors.Is(err, litertlm.ErrUnimplemented) {
			fmt.Fprintf(os.Stderr, "WithDebuggerEnabled requires LiteRT-LM C API 1.1.0+ (v0.20.0): %v\n", err)
			fmt.Fprintln(os.Stderr, "Falling back to non-debugger session to demonstrate capture-dir inspection.")
			client, err = litertlm.New(ctx,
				litertlm.WithLib(resolvedLib),
				litertlm.WithModel(resolvedModel),
				litertlm.WithBackend(*backend),
				litertlm.WithCacheDir(*cacheDir),
			)
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "new client: %v\n", err)
			os.Exit(1)
		}
	}
	defer func() { _ = client.Close() }()

	chat, err := client.NewChat(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new chat: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = chat.Close() }()

	reply, err := chat.Send(ctx, *prompt, litertlm.WithMaxOutputTokens(32))
	if err != nil {
		fmt.Fprintf(os.Stderr, "chat send: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Chat reply: %s\n", reply.Text())

	chatDir, err := chat.DebugCaptureDir()
	if err != nil {
		fmt.Fprintf(os.Stderr, "chat debug dir: %v\n", err)
		os.Exit(1)
	}
	printCaptureDir("Chat", *cacheDir, chatDir)

	sess, err := client.Engine().NewSession(0)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new session: %v\n", err)
		os.Exit(1)
	}
	defer sess.Delete()

	sessDir, err := sess.DebugCaptureDir()
	if err != nil {
		fmt.Fprintf(os.Stderr, "session debug dir: %v\n", err)
		os.Exit(1)
	}
	printCaptureDir("Session", *cacheDir, sessDir)
}

func printCaptureDir(kind, cacheDir, rel string) {
	if rel == "" {
		fmt.Printf("%s capture dir: (disabled or unsupported by runtime)\n", kind)
		return
	}
	fmt.Printf("%s capture dir: %s (remember to delete unencrypted trace files when finished)\n",
		kind, filepath.Join(cacheDir, rel))
}
