package litertlm

import "context"

// runCancellable runs work in a goroutine and waits for either work to
// complete or ctx to be cancelled. On ctx cancellation, it spawns a
// background goroutine that disposes work's eventual successful result
// via cleanup; the caller receives ctx.Err() promptly.
//
// The C calls wrapped by runCancellable (Load, NewEngine,
// NewConversation) are synchronous and have no C-side cancel hook —
// once started they run to completion. This helper lets a Go caller
// return on ctx cancellation without leaking the handle the C call
// eventually produces. cleanup is invoked exactly once if and only if
// work returns a nil error after the caller has already given up.
//
// If ctx is already cancelled at entry, work is not started and
// ctx.Err() is returned immediately.
func runCancellable[T any](
	ctx context.Context,
	work func() (T, error),
	cleanup func(T),
) (T, error) {
	var zero T
	if err := ctx.Err(); err != nil {
		return zero, err
	}

	type res struct {
		v   T
		err error
	}
	done := make(chan res, 1)
	go func() {
		v, err := work()
		done <- res{v: v, err: err}
	}()

	select {
	case <-ctx.Done():
		go func() {
			r := <-done
			if r.err == nil && cleanup != nil {
				cleanup(r.v)
			}
		}()
		return zero, ctx.Err()
	case r := <-done:
		return r.v, r.err
	}
}
