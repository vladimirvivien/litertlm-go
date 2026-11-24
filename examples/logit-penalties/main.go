// logit-penalties demonstrates per-turn logit shaping using
// WithRepetitionPenalty, WithNoRepeatNgram, and WithSuppressTokens.
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

	toks, err := client.Tokenize("blue")
	if err != nil {
		fmt.Fprintf(os.Stderr, "tokenize: %v\n", err)
		os.Exit(1)
	}
	suppressIDs := make([]int, len(toks))
	for i, id := range toks {
		suppressIDs[i] = int(id)
	}

	chat, err := client.NewChat(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new chat: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = chat.Close() }()

	fmt.Println("=== WithRepetitionPenalty + WithNoRepeatNgram ===")
	reply1, err := chat.Send(ctx,
		"Name three primary colors and describe each briefly.",
		litertlm.WithMaxOutputTokens(64),
		litertlm.WithRepetitionPenalty(litertlm.RepetitionPenaltyConfig{
			RepetitionPenalty: 1.15,
			PresencePenalty:   0.2,
			FrequencyPenalty:  0.1,
			WindowSize:        64,
		}),
		litertlm.WithNoRepeatNgram(litertlm.NoRepeatNgramConfig{
			NgramSize:  3,
			WindowSize: 64,
		}),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "repetition/ngram send: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(reply1.Text())

	fmt.Printf("=== WithSuppressTokens (suppressing token IDs %v for \"blue\") ===\n", suppressIDs)
	chat2, err := client.NewChat(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "new chat 2: %v\n", err)
		os.Exit(1)
	}
	defer func() { _ = chat2.Close() }()

	reply2, err := chat2.Send(ctx,
		"Name three colors of the rainbow.",
		litertlm.WithMaxOutputTokens(48),
		litertlm.WithSuppressTokens(suppressIDs...),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "suppress tokens send: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(reply2.Text())
}
