package litertlm

import (
	"context"
	"fmt"
	"iter"
)

// Chunk is one piece of a streaming Generate response.
type Chunk struct {
	Text  string
	Final bool
}

// Generate runs synchronous inference for prompt and returns the first
// candidate's text. Cancelling ctx aborts the in-flight inference (via
// Session.Cancel) and returns ctx.Err.
func (c *Client) Generate(ctx context.Context, prompt string, opts ...GenOption) (string, error) {
	sess, err := c.openSession(opts)
	if err != nil {
		return "", err
	}
	defer sess.Delete()

	stop := wireCancel(ctx, sess.Cancel)
	defer stop()

	resp, err := sess.GenerateContent([]InputData{NewTextInputString(prompt)})
	if err != nil {
		if cerr := ctx.Err(); cerr != nil {
			return "", cerr
		}
		return "", err
	}
	defer resp.Delete()

	if resp.NumCandidates() == 0 {
		return "", fmt.Errorf("litertlm: Generate: no candidates returned")
	}
	return resp.Text(0), nil
}

// GenerateStream returns an iterator over response chunks. Cancelling
// ctx aborts the stream via Session.Cancel; the iterator yields the
// surfaced error and closes.
//
//	for chunk, err := range client.GenerateStream(ctx, prompt) {
//	    if err != nil { return err }
//	    fmt.Print(chunk.Text)
//	}
func (c *Client) GenerateStream(ctx context.Context, prompt string, opts ...GenOption) iter.Seq2[Chunk, error] {
	return func(yield func(Chunk, error) bool) {
		sess, err := c.openSession(opts)
		if err != nil {
			yield(Chunk{}, err)
			return
		}
		defer sess.Delete()

		stop := wireCancel(ctx, sess.Cancel)
		defer stop()

		for sc := range sess.GenerateContentStreamCh([]InputData{NewTextInputString(prompt)}) {
			ch := Chunk{Text: sc.Text, Final: sc.Final}
			if !yield(ch, sc.Err) {
				return
			}
			if sc.Err != nil {
				return
			}
		}
	}
}

// wireCancel arranges for ctx cancellation to call cancelFn (typically
// Session.Cancel or Conversation.Cancel). Returns a stop function the
// caller defers to release the watcher goroutine. No-op (returns a
// no-op stop) when ctx cannot be cancelled.
func wireCancel(ctx context.Context, cancelFn func()) (stop func()) {
	if ctx == nil || ctx.Done() == nil {
		return func() {}
	}
	done := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			cancelFn()
		case <-done:
		}
	}()
	return func() { close(done) }
}
