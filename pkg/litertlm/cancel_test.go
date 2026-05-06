package litertlm

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

// TestRunCancellable_NormalCompletion: when ctx never cancels, the
// helper returns whatever work returns.
func TestRunCancellable_NormalCompletion(t *testing.T) {
	ctx := context.Background()
	v, err := runCancellable(ctx,
		func() (int, error) { return 42, nil },
		func(int) { t.Fatal("cleanup must not run on success") },
	)
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if v != 42 {
		t.Errorf("v = %d, want 42", v)
	}
}

// TestRunCancellable_WorkError: errors from work are propagated.
func TestRunCancellable_WorkError(t *testing.T) {
	wantErr := errors.New("boom")
	_, err := runCancellable(context.Background(),
		func() (int, error) { return 0, wantErr },
		func(int) { t.Fatal("cleanup must not run on error") },
	)
	if !errors.Is(err, wantErr) {
		t.Fatalf("err = %v, want %v", err, wantErr)
	}
}

// TestRunCancellable_PreCancelled: a context cancelled before entry
// short-circuits without ever invoking work.
func TestRunCancellable_PreCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var ran atomic.Bool
	_, err := runCancellable(ctx,
		func() (int, error) { ran.Store(true); return 0, nil },
		nil,
	)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if ran.Load() {
		t.Error("work must not run when ctx is already cancelled")
	}
}

// TestRunCancellable_CancelDuringWork: ctx cancellation while work is
// in flight returns ctx.Err() to the caller and runs cleanup once
// work eventually succeeds in the background.
func TestRunCancellable_CancelDuringWork(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	workDone := make(chan struct{})
	cleanupDone := make(chan int, 1)

	go func() {
		// Give runCancellable a moment to enter the select, then cancel.
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()

	_, err := runCancellable(ctx,
		func() (int, error) {
			<-workDone
			return 99, nil
		},
		func(v int) { cleanupDone <- v },
	)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}

	// Now release the in-flight work; cleanup must run with the value.
	close(workDone)
	select {
	case got := <-cleanupDone:
		if got != 99 {
			t.Errorf("cleanup got %d, want 99", got)
		}
	case <-time.After(time.Second):
		t.Fatal("cleanup did not run within 1s")
	}
}

// TestRunCancellable_CancelDuringWork_WithError: ctx cancels while
// work is in flight and work then fails — cleanup must NOT run.
func TestRunCancellable_CancelDuringWork_WithError(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	workDone := make(chan struct{})
	var cleanupRan atomic.Bool

	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()

	_, err := runCancellable(ctx,
		func() (int, error) {
			<-workDone
			return 0, errors.New("late failure")
		},
		func(int) { cleanupRan.Store(true) },
	)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}

	close(workDone)
	// Give the background cleanup goroutine time to run (it shouldn't).
	time.Sleep(50 * time.Millisecond)
	if cleanupRan.Load() {
		t.Error("cleanup ran on errored work; must only run on success")
	}
}
