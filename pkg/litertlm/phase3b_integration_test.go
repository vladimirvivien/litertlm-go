package litertlm_test

import (
	"context"
	"errors"
	"io"
	"testing"

	"github.com/vladimirvivien/litertlm-go/pkg/litertlm"
)

func TestLive_Phase3b_ExperimentalAndVersionGates(t *testing.T) {
	libDir, modelPath := requireTestModel(t)
	ctx := context.Background()
	litertlm.SetMinLogLevel(litertlm.LogQuiet)

	client, err := litertlm.New(ctx,
		litertlm.WithLib(libDir),
		litertlm.WithModel(modelPath),
		litertlm.WithBackend("cpu"),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = client.Close() }()

	if err = client.SetMetalResidency(true); err != nil {
		t.Fatalf("SetMetalResidency(true): %v", err)
	}
	if err = client.SetMetalResidency(false); err != nil {
		t.Fatalf("SetMetalResidency(false): %v", err)
	}

	chat, err := client.NewChat(ctx)
	if err != nil {
		t.Fatalf("NewChat: %v", err)
	}
	defer func() { _ = chat.Close() }()

	chatDir, err := chat.DebugCaptureDir()
	if err != nil {
		t.Fatalf("Chat.DebugCaptureDir: %v", err)
	}
	if chatDir != "" {
		t.Errorf("expected empty Chat.DebugCaptureDir when debugger disabled, got %q", chatDir)
	}

	sess, err := client.Engine().NewSession(0)
	if err != nil {
		t.Fatalf("NewSession: %v", err)
	}
	defer sess.Delete()

	sessDir, err := sess.DebugCaptureDir()
	if err != nil {
		t.Fatalf("Session.DebugCaptureDir: %v", err)
	}
	if sessDir != "" {
		t.Errorf("expected empty Session.DebugCaptureDir when debugger disabled, got %q", sessDir)
	}

	_, minorVer, _, verErr := litertlm.APIVersion()
	if verErr != nil {
		t.Fatalf("APIVersion: %v", verErr)
	}
	if minorVer < 1 {
		assertCUnimplemented := func(op string, callErr error) {
			t.Helper()
			if !errors.Is(callErr, litertlm.ErrUnimplemented) {
				t.Errorf("%s err = %v, want ErrUnimplemented on C API 1.0.x", op, callErr)
			}
			if _, ok := errors.AsType[*litertlm.CError](callErr); !ok {
				t.Errorf("%s err type = %T, want *CError", op, callErr)
			}
		}

		assertCUnimplemented("SetEnableDebugger", client.Settings().SetEnableDebugger(true))

		dummyFetcher := litertlm.RangeFetcherFunc(func(context.Context, uint64, uint64, io.Writer) error { return nil })
		_, planErr := litertlm.PlanModelSync(ctx, modelPath, dummyFetcher)
		assertCUnimplemented("PlanModelSync", planErr)

		syncErr := litertlm.SyncModel(ctx, modelPath, modelPath+".tmp", dummyFetcher)
		assertCUnimplemented("SyncModel", syncErr)

		_, cacheErr := litertlm.SyncModelCache(ctx, t.TempDir(), "test-model", dummyFetcher)
		assertCUnimplemented("SyncModelCache", cacheErr)
	}
}
