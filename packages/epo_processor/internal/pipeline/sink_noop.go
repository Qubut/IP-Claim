package pipeline

import "context"

// NoopSink discards record batches. Pair with NoopExtractor when the
// pipeline's job is only to persist archives / extracted files to disk.
type NoopSink struct{}

// Write discards the batch.
func (NoopSink) Write(_ context.Context, _ []PatentRecord) error { return nil }

// Close is a no-op.
func (NoopSink) Close() error { return nil }
