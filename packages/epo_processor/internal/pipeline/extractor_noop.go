package pipeline

import (
	"context"
	"io"

	"github.com/destel/rill"
)

// NoopExtractor reads each entry to EOF and closes it without producing
// any PatentRecord. Use it when the goal is on-disk extraction only:
// reading the entry is what flushes any TeeReader installed by the walker.
type NoopExtractor struct{}

// Stream drains the entry to EOF without emitting any records.
func (NoopExtractor) Stream(_ context.Context, e XMLEntry) rill.Stream[PatentRecord] {
	return rill.Generate(func(_ func(PatentRecord), sendErr func(error)) {
		defer func() { _ = e.Close() }()
		if _, err := io.Copy(io.Discard, e.Reader); err != nil {
			sendErr(err)
		}
	})
}
