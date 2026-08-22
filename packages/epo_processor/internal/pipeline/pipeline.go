package pipeline

import (
	"context"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/destel/rill"
)

// Pipeline runs the configured stages end-to-end.
type Pipeline struct{ opts Options }

// New constructs a Pipeline from functional options.
func New(opts ...Option) (*Pipeline, error) {
	var o Options
	for _, fn := range opts {
		fn(&o)
	}
	o.defaults()
	if err := o.validate(); err != nil {
		return nil, err
	}
	return &Pipeline{opts: o}, nil
}

// Run executes the pipeline and blocks until the source is exhausted,
// the first stage error occurs, or ctx is cancelled.
func (p *Pipeline) Run(ctx context.Context) (runErr error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Always close the sink; do not mask a pipeline error with a close error.
	defer func() {
		if cerr := p.opts.Sink.Close(); cerr != nil && runErr == nil {
			runErr = fmt.Errorf("sink close: %w", cerr)
		}
	}()

	log := p.opts.Logger
	notify := p.opts.Progress
	startedAt := time.Now()
	var (
		archivesSeen   atomic.Int64
		entriesSeen    atomic.Int64
		recordsWritten atomic.Int64
		batchesWritten atomic.Int64
	)
	emit := func() {
		notify(Stats{
			Archives: archivesSeen.Load(),
			Entries:  entriesSeen.Load(),
			Records:  recordsWritten.Load(),
			Batches:  batchesWritten.Load(),
		})
	}

	jobs := p.opts.Source.Stream(ctx)

	jobs = rill.OrderedMap(jobs, 1, func(j ArchiveJob) (ArchiveJob, error) {
		n := archivesSeen.Add(1)
		log.Info("archive: queued", "n", n, "label", j.Name, "url", j.URL)
		emit()
		return j, nil
	})

	// Unordered FlatMap so every worker drains its own archive stream
	// independently.
	entries := rill.FlatMap(jobs, p.opts.ArchiveConcurrency,
		func(j ArchiveJob) <-chan rill.Try[XMLEntry] {
			log.Info("archive: opening", "label", j.Name, "url", j.URL, "local_path", j.LocalPath)
			return p.opts.Opener.Stream(ctx, j)
		})

	entries = rill.OrderedMap(entries, 1, func(e XMLEntry) (XMLEntry, error) {
		entriesSeen.Add(1)
		emit()
		return e, nil
	})

	// Extractor owns each entry's lifetime and must call entry.Close().
	// Unordered FlatMap so ExtractorConcurrency parsers run in parallel.
	records := rill.FlatMap(entries, p.opts.ExtractorConcurrency,
		func(e XMLEntry) <-chan rill.Try[PatentRecord] {
			return p.opts.Extractor.Stream(ctx, e)
		})

	// Safety net: downgrade any residual per-entry / parse stream error to
	// a warning and drop it, so no single record aborts the whole run.
	records = rill.Catch(records, 1, func(err error) error {
		log.Warn("pipeline: skipping record-stream error (continuing)", "err", err)
		return nil
	})

	batches := rill.Batch(records, p.opts.BatchSize, p.opts.BatchTimeout)

	// Single writer: sink need not be goroutine-safe.
	err := rill.ForEach(batches, 1, func(batch []PatentRecord) error {
		if len(batch) == 0 {
			return nil
		}
		if err := p.opts.Sink.Write(ctx, batch); err != nil {
			return err
		}
		bn := batchesWritten.Add(1)
		recordsWritten.Add(int64(len(batch)))
		emit()
		if bn%10 == 1 {
			log.Info("sink: batch written",
				"batch", bn,
				"size", len(batch),
				"records_total", recordsWritten.Load(),
				"entries_total", entriesSeen.Load(),
			)
		}
		return nil
	})

	emit()
	log.Info("pipeline: done",
		"archives", archivesSeen.Load(),
		"entries", entriesSeen.Load(),
		"records", recordsWritten.Load(),
		"batches", batchesWritten.Load(),
		"elapsed", time.Since(startedAt).String(),
		"err", err,
	)
	return err
}
