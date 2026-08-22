package pipeline

import (
	"errors"
	"log/slog"
	"time"
)

// Stats is a monotonically-increasing snapshot of pipeline counters.
// All fields are updated atomically and safe to read from any goroutine.
type Stats struct {
	Archives int64 // total archives fully drained
	Entries  int64 // total XML entries emitted across all archives
	Records  int64 // total PatentRecord values produced by extractors
	Batches  int64 // total Write calls delivered to the sink
}

// ProgressFunc receives a Stats snapshot on every counter update.
// Implementations must be cheap and goroutine-safe.
type ProgressFunc func(Stats)

// Options configures a Pipeline. Use WithXxx helpers to populate it.
// Required fields: Source, Opener, Extractor, Sink.
// All other fields have safe defaults (see WithXxx documentation).
type Options struct {
	// Source produces the stream of ArchiveJob values to process.
	Source ArchiveSource
	// Opener fetches or opens each archive and emits XMLEntry values.
	Opener ArchiveOpener
	// Extractor parses each XMLEntry into PatentRecord values.
	Extractor RecordExtractor
	// Sink consumes batched PatentRecord values (serialised, one goroutine).
	Sink RecordSink
	// Janitor receives per-archive and per-entry lifecycle notifications.
	// Defaults to NoopJanitor.
	Janitor Janitor
	// Logger receives structured pipeline events. Defaults to a discard logger.
	Logger *slog.Logger
	// Progress is called on every counter update. Defaults to no-op.
	Progress ProgressFunc
	// ArchiveConcurrency is the number of archives in flight simultaneously.
	// Default: 4.
	ArchiveConcurrency int
	// ExtractorConcurrency is the number of in-flight entries decoded in
	// parallel (one sequential, CPU-bound reader each) — the knob that
	// saturates cores. Auto-sizing (NumCPU clamped by a memory budget) is a
	// caller policy: resolve it with PlanReaderConcurrency before passing it
	// here. Values <= 0 fall back to a minimal safe default. Default: 4.
	ExtractorConcurrency int
	// BatchSize is the target number of records per sink Write call.
	// Default: 1000.
	BatchSize int
	// BatchTimeout is the maximum idle period before a partial batch is
	// flushed to the sink. Default: 2s.
	BatchTimeout time.Duration
}

// Option mutates Options.
type Option func(*Options)

// WithSource sets the archive job source. Required.
func WithSource(s ArchiveSource) Option { return func(o *Options) { o.Source = s } }

// WithOpener sets the archive opener. Required.
func WithOpener(o ArchiveOpener) Option { return func(opts *Options) { opts.Opener = o } }

// WithExtractor sets the XML record extractor. Required.
func WithExtractor(e RecordExtractor) Option { return func(o *Options) { o.Extractor = e } }

// WithSink sets the record sink. Required.
func WithSink(s RecordSink) Option { return func(o *Options) { o.Sink = s } }

// WithJanitor sets the post-archive janitor.
// Default: [NoopJanitor].
func WithJanitor(j Janitor) Option { return func(o *Options) { o.Janitor = j } }

// WithLogger sets the structured logger.
// Default: a discard logger (no output).
func WithLogger(l *slog.Logger) Option { return func(o *Options) { o.Logger = l } }

// WithProgress sets the progress callback, called on every counter update.
// Default: no-op.
func WithProgress(f ProgressFunc) Option { return func(o *Options) { o.Progress = f } }

// WithArchiveConcurrency sets the number of archives processed in parallel.
// Values ≤ 0 are treated as 4.
func WithArchiveConcurrency(n int) Option { return func(o *Options) { o.ArchiveConcurrency = n } }

// WithExtractorConcurrency sets the number of in-flight entries decoded in
// parallel. Auto-sizing is a caller concern (see PlanReaderConcurrency); this
// only applies a minimal safe fallback for values ≤ 0.
func WithExtractorConcurrency(n int) Option {
	return func(o *Options) { o.ExtractorConcurrency = n }
}

// WithBatchSize sets the target number of records per sink Write call.
// Values ≤ 0 are treated as 1000.
func WithBatchSize(n int) Option { return func(o *Options) { o.BatchSize = n } }

// WithBatchTimeout sets the idle flush period for partial batches.
// A zero value is treated as 2s.
func WithBatchTimeout(d time.Duration) Option { return func(o *Options) { o.BatchTimeout = d } }

// defaults fills in zero-valued fields with safe defaults.
func (o *Options) defaults() {
	if o.ArchiveConcurrency <= 0 {
		o.ArchiveConcurrency = 4
	}
	if o.ExtractorConcurrency <= 0 {
		o.ExtractorConcurrency = 4
	}
	if o.BatchSize <= 0 {
		o.BatchSize = 1000
	}
	if o.BatchTimeout == 0 {
		o.BatchTimeout = 2 * time.Second
	}
	if o.Janitor == nil {
		o.Janitor = NoopJanitor{}
	}
	if o.Logger == nil {
		o.Logger = slog.New(slog.DiscardHandler)
	}
	if o.Progress == nil {
		o.Progress = func(Stats) {}
	}
}

// validate returns an error when a required dependency is missing.
func (o *Options) validate() error {
	switch {
	case o.Source == nil:
		return errors.New("pipeline: Source is required (use WithSource)")
	case o.Opener == nil:
		return errors.New("pipeline: Opener is required (use WithOpener)")
	case o.Extractor == nil:
		return errors.New("pipeline: Extractor is required (use WithExtractor)")
	case o.Sink == nil:
		return errors.New("pipeline: Sink is required (use WithSink)")
	}
	return nil
}
