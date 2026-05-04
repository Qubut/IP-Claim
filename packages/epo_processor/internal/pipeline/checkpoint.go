package pipeline

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
	"time"

	O "github.com/IBM/fp-go/v2/ord"
	"github.com/IBM/fp-go/v2/record"
	"github.com/destel/rill"
	bolt "go.etcd.io/bbolt"
	"go.uber.org/zap"
)

// whenLog calls fn with log when log is non-nil.
func whenLog(log *zap.SugaredLogger, fn func(*zap.SugaredLogger)) {
	if log != nil {
		fn(log)
	}
}

// Checkpointer records which archive jobs have been fully processed so
// they can be skipped on resume. Implementations must be goroutine-safe.
type Checkpointer interface {
	// IsCompleted reports whether the job with the given stable ID has
	// already been processed in a previous run.
	IsCompleted(jobID string) (bool, error)
	// MarkCompleted atomically and durably records that the job has been
	// fully drained and its records flushed to the sink.
	MarkCompleted(jobID string, meta map[string]string) error
	// HasAny reports whether at least one job has been recorded as completed.
	HasAny() (bool, error)
	// Close releases any underlying resources.
	Close() error
}

// jobID returns a stable 128-bit hex digest derived from the job's URL,
// LocalPath or Name (in that order). Suitable as a bbolt bucket key.
func jobID(j ArchiveJob) string {
	key := j.URL
	if key == "" {
		key = j.LocalPath
	}
	if key == "" {
		key = j.Name
	}
	sum := sha256.Sum256([]byte(key))
	return hex.EncodeToString(sum[:16])
}

// ----------------------------------------------------------------------------
// Null Object — used when checkpointing is disabled.
// ----------------------------------------------------------------------------

// NoopCheckpointer is a no-op Checkpointer used when the feature is
// disabled. IsCompleted always returns false; MarkCompleted is a no-op.
type NoopCheckpointer struct{}

// IsCompleted always returns false for a no-op checkpointer.
func (NoopCheckpointer) IsCompleted(string) (bool, error) { return false, nil }

// MarkCompleted is a no-op.
func (NoopCheckpointer) MarkCompleted(string, map[string]string) error { return nil }

// HasAny always returns false for a no-op checkpointer.
func (NoopCheckpointer) HasAny() (bool, error) { return false, nil }

// Close is a no-op.
func (NoopCheckpointer) Close() error { return nil }

// ----------------------------------------------------------------------------
// Repository — bbolt-backed implementation.
// ----------------------------------------------------------------------------

var checkpointBucket = []byte("completed_jobs")

// BoltCheckpointer is an embedded, ACID, single-file Checkpointer backed
// by bbolt (https://github.com/etcd-io/bbolt). Goroutine-safe.
type BoltCheckpointer struct {
	db   *bolt.DB
	path string
	mu   sync.Mutex
}

// OpenBoltCheckpointer opens or creates a checkpoint database file at
// path. The caller must Close it.
func OpenBoltCheckpointer(path string) (*BoltCheckpointer, error) {
	db, err := bolt.Open(path, 0o600, &bolt.Options{Timeout: 2 * time.Second})
	if err != nil {
		return nil, fmt.Errorf("open checkpoint %s: %w", path, err)
	}
	if err := db.Update(func(tx *bolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists(checkpointBucket)
		return err
	}); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("init checkpoint bucket: %w", err)
	}
	return &BoltCheckpointer{db: db, path: path}, nil
}

// IsCompleted reports whether id has been marked completed.
func (b *BoltCheckpointer) IsCompleted(id string) (bool, error) {
	var found bool
	err := b.db.View(func(tx *bolt.Tx) error {
		found = tx.Bucket(checkpointBucket).Get([]byte(id)) != nil
		return nil
	})
	return found, err
}

// MarkCompleted persists id and its metadata as a completed entry.
func (b *BoltCheckpointer) MarkCompleted(id string, meta map[string]string) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	// Encode meta as compact `ts;k=v;k=v`. Key order is deterministic via
	// the supplied [ord.Ord], so checkpoint files are diff-friendly.
	val := record.ReduceOrdWithIndex[string, string](O.FromStrictCompare[string]())(
		func(k, acc, v string) string { return acc + ";" + k + "=" + v },
		time.Now().UTC().Format(time.RFC3339),
	)(meta)
	return b.db.Update(func(tx *bolt.Tx) error {
		return tx.Bucket(checkpointBucket).Put([]byte(id), []byte(val))
	})
}

// Close flushes and closes the underlying bbolt database.
func (b *BoltCheckpointer) Close() error { return b.db.Close() }
// HasAny returns true when the checkpoint bucket contains at least one entry.
func (b *BoltCheckpointer) HasAny() (bool, error) {
	var found bool
	err := b.db.View(func(tx *bolt.Tx) error {
		k, _ := tx.Bucket(checkpointBucket).Cursor().First()
		found = k != nil
		return nil
	})
	return found, err
}

// ----------------------------------------------------------------------------
// Decorator — wraps an ArchiveSource to filter completed jobs.
// ----------------------------------------------------------------------------

// CheckpointSource decorates an ArchiveSource, filtering out jobs whose
// stable ID is already marked completed by CP. Read errors from CP are
// logged and the job is forwarded (best-effort — prefer reprocess to halt).
type CheckpointSource struct {
	Inner  ArchiveSource
	CP     Checkpointer
	Logger *zap.SugaredLogger
}

// NewCheckpointSource constructs a CheckpointSource.
func NewCheckpointSource(inner ArchiveSource, cp Checkpointer, log *zap.SugaredLogger) *CheckpointSource {
	return &CheckpointSource{Inner: inner, CP: cp, Logger: log}
}

// Stream returns the filtered job channel, skipping already-completed archives.
func (c *CheckpointSource) Stream(ctx context.Context) <-chan rill.Try[ArchiveJob] {
	return rill.Filter(c.Inner.Stream(ctx), 1, func(j ArchiveJob) (bool, error) {
		done, err := c.CP.IsCompleted(jobID(j))
		if err != nil {
			whenLog(c.Logger, func(l *zap.SugaredLogger) {
				l.Warnw("checkpoint: read failed, will reprocess", "name", j.Name, "err", err)
			})
			return true, nil
		}
		if done {
			whenLog(c.Logger, func(l *zap.SugaredLogger) {
				l.Infow("checkpoint: skipping completed", "name", j.Name)
			})
			return false, nil
		}
		return true, nil
	})
}

// ----------------------------------------------------------------------------
// Janitor adapter — marks archives complete via the existing opener hook.
// ----------------------------------------------------------------------------

// CheckpointJanitor marks an archive completed in CP after the opener
// finishes draining it without error.
//
// Completion means all entries were enqueued downstream, not that every
// record was persisted. With a single-file Parquet sink an interrupted
// run may leave a truncated file; on resume the cmd layer routes new
// output to a fresh shard so previous output is preserved.
type CheckpointJanitor struct {
	CP     Checkpointer
	Logger *zap.SugaredLogger
}

// OnArchiveDone marks j completed in CP after the opener finishes draining it.
func (c CheckpointJanitor) OnArchiveDone(_ context.Context, j ArchiveJob, _ string, _ bool) {
	if err := c.CP.MarkCompleted(jobID(j), map[string]string{"name": j.Name}); err != nil {
		whenLog(c.Logger, func(l *zap.SugaredLogger) {
			l.Warnw("checkpoint: mark failed", "name", j.Name, "err", err)
		})
		return
	}
	whenLog(c.Logger, func(l *zap.SugaredLogger) {
		l.Infow("checkpoint: archive marked complete", "name", j.Name)
	})
}

// OnEntryDone is a no-op; entry-level checkpointing is not required.
func (CheckpointJanitor) OnEntryDone(context.Context, XMLEntry, string) {}
