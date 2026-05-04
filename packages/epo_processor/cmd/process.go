package cmd

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/IBM/fp-go/v2/array"
	F "github.com/IBM/fp-go/v2/function"
	"github.com/mattn/go-isatty"
	"github.com/parquet-go/parquet-go"
	"github.com/schollz/progressbar/v3"
	"github.com/spf13/cobra"
	"go.uber.org/zap"

	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/pipeline"
)

// progressBar abstracts the TUI progress bar so the command code never
// has to nil-check it. Real bar in TTY, noopProgressBar otherwise.
type progressBar interface {
	Describe(string)
	Set64(int64) error
	Finish() error
}

// noopProgressBar is the silent fallback used when stdout is not a TTY.
type noopProgressBar struct{}

func (noopProgressBar) Describe(string)   {}
func (noopProgressBar) Set64(int64) error { return nil }
func (noopProgressBar) Finish() error     { return nil }

// newProgressBar returns a TUI bar when stdout is a TTY, noopProgressBar
// otherwise. Always non-nil.
func newProgressBar(label string) progressBar {
	if !isatty.IsTerminal(os.Stdout.Fd()) {
		return noopProgressBar{}
	}
	return progressbar.NewOptions(-1,
		progressbar.OptionSetDescription(label),
		progressbar.OptionSetWriter(os.Stdout),
		progressbar.OptionSpinnerType(14),
		progressbar.OptionSetElapsedTime(true),
		progressbar.OptionShowIts(),
		progressbar.OptionSetItsString("rec"),
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionUseANSICodes(true),
		progressbar.OptionThrottle(80*time.Millisecond),
		progressbar.OptionClearOnFinish(),
	)
}

// processCmd runs the streaming ETL: list product, fetch each archive,
// recursively unwrap, parse XML, write Parquet.
var processCmd = &cobra.Command{
	Use:   "process",
	Short: "Stream EPO archives end-to-end into Parquet (download → extract → parse)",
		RunE: func(_ *cobra.Command, _ []string) error {
		ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer cancel()

		source, err := buildSource()
		if err != nil {
			return err
		}
		opener, err := buildOpener()
		if err != nil {
			return err
		}

		// Optional resumability — opt-in via pipeline.checkpoint_db.
		// Honour reset-checkpoint *before* opening so we delete the bbolt
		// file rather than racing against an in-process handle.
		if cfg.Pipeline.ResetCheckpoint {
			if err := resetState(cfg.Pipeline.CheckpointDB, cfg.Pipeline.OutputParquet); err != nil {
				return fmt.Errorf("reset-checkpoint: %w", err)
			}
		}
		cp, closeCP, err := buildCheckpointer(cfg.Pipeline.CheckpointDB)
		if err != nil {
			return err
		}
		defer func() { _ = closeCP() }()

		// Resolve the output path: append rather than overwrite by writing
		// to a fresh shard whenever the checkpoint already records work.
		// Parquet has no in-place append (footer at EOF), so each resumed
		// run produces <stem>.part-<unix>.parquet next to the original.
		resume := false
		if _, isNoop := cp.(pipeline.NoopCheckpointer); !isNoop {
			hasWork, herr := cp.HasAny()
			if herr != nil {
				return fmt.Errorf("checkpoint: probe state: %w", herr)
			}
			resume = hasWork
		}
		outPath := resolveOutputPath(cfg.Pipeline.OutputParquet, resume)
		if resume && outPath != cfg.Pipeline.OutputParquet {
			logger.Warnw("resume: writing to a new shard to preserve previous output",
				"requested", cfg.Pipeline.OutputParquet, "actual", outPath)
		}

		sink, err := pipeline.NewParquetSink(outPath, cfg.Pipeline.RowGroupSize)
		if err != nil {
			return err
		}
		defer func() { _ = sink.Close() }()

		if _, ok := cp.(pipeline.NoopCheckpointer); !ok {
			source = pipeline.NewCheckpointSource(source, cp, logger)
			attachCheckpointJanitor(opener, cp)
			logger.Infow("checkpoint: enabled", "db", cfg.Pipeline.CheckpointDB, "resume", resume)
		}

		bar := newProgressBar("[cyan]streaming EPO[reset]")
		var (
			barMu      sync.Mutex
			lastStats  pipeline.Stats
			curArchive string
			curDone    int64
			curTotal   int64
		)
		describe := func() {
			bar.Describe(fmt.Sprintf(
				"[cyan]archives[reset] %d  [magenta]entries[reset] %d  [green]records[reset] %d  [yellow]batches[reset] %d  [white]%s %s[reset]",
				lastStats.Archives, lastStats.Entries, lastStats.Records, lastStats.Batches,
				curArchive, fmtBytes(curDone, curTotal),
			))
		}
		opts := []pipeline.Option{
			pipeline.WithSource(source),
			pipeline.WithOpener(opener),
			pipeline.WithExtractor(pipeline.NewXMLStreamExtractor()),
			pipeline.WithSink(sink),
			pipeline.WithLogger(logger),
			pipeline.WithArchiveConcurrency(cfg.Pipeline.ArchiveConcurrency),
			pipeline.WithExtractorConcurrency(cfg.Pipeline.ExtractorConcurrency),
			pipeline.WithBatchSize(cfg.Pipeline.BatchSize),
			pipeline.WithBatchTimeout(cfg.Pipeline.BatchTimeout),
		}
		opts = append(opts, pipeline.WithProgress(func(s pipeline.Stats) {
			barMu.Lock()
			defer barMu.Unlock()
			lastStats = s
			describe()
			_ = bar.Set64(s.Records)
		}))
		if h, ok := opener.(*pipeline.HTTPOpener); ok {
			var lastLogged time.Time
			h.OnBytes = func(job pipeline.ArchiveJob, downloaded, total int64) {
				barMu.Lock()
				defer barMu.Unlock()
				curArchive, curDone, curTotal = job.Name, downloaded, total
				describe()
				if now := time.Now(); now.Sub(lastLogged) >= 5*time.Second {
					lastLogged = now
					logger.Infow("download: progress", "label", job.Name, "bytes", fmtBytes(downloaded, total))
				}
			}
		}

		p, err := pipeline.New(opts...)
		if err != nil {
			return fmt.Errorf("build pipeline: %w", err)
		}

		logger.Infow("Starting streaming pipeline",
			"output", outPath,
			"archive_concurrency", cfg.Pipeline.ArchiveConcurrency,
			"extractor_concurrency", cfg.Pipeline.ExtractorConcurrency,
			"batch_size", cfg.Pipeline.BatchSize,
		)
		if err := p.Run(ctx); err != nil {
			_ = bar.Finish()
			return fmt.Errorf("pipeline: %w", err)
		}
		_ = bar.Finish()
		logger.Info("Streaming pipeline completed")

		// Merge shards into the main output file so the workspace stays tidy
		// and downstream tools (analyze) always read a single file.
		if err := mergeParquetShards(cfg.Pipeline.OutputParquet, logger); err != nil {
			logger.Warnw("merge shards: failed (non-fatal)", "err", err)
		}
		return nil
	},
}

// buildSource selects between live EPO listing and a local-directory replay.
func buildSource() (pipeline.ArchiveSource, error) {
	if dir := strings.TrimSpace(cfg.Pipeline.UseLocalDir); dir != "" {
		jobs, err := localJobsFromDir(dir)
		if err != nil {
			return nil, fmt.Errorf("scan %s: %w", dir, err)
		}
		return &pipeline.StaticListSource{Jobs: jobs}, nil
	}
	client := &http.Client{Timeout: cfg.Server.Timeout}
	return pipeline.NewEPOProductSource(cfg.Server.BaseURL, cfg.Server.ProductID, client), nil
}

// buildOpener picks the matching opener for the chosen source.
func buildOpener() (pipeline.ArchiveOpener, error) {
	walk := walkConfigFromCfg()
	if strings.TrimSpace(cfg.Pipeline.UseLocalDir) != "" {
		l := pipeline.NewLocalFileOpener(walk)
		l.Logger = logger
		return l, nil
	}
	timeout := cfg.Server.Timeout
	if timeout < 1*time.Minute {
		timeout = 0 // unlimited for streaming downloads
	}
	client := &http.Client{Timeout: timeout}
	o := pipeline.NewHTTPOpener(
		client,
		uint(cfg.Server.MaxRetries), //nolint:gosec // MaxRetries is small, overflow impossible
		cfg.Server.VerifySHA1,
		walk,
	)
	o.KeepArchive = cfg.Pipeline.KeepArchive
	o.ArchiveDir = cfg.Pipeline.ArchiveDir
	o.Logger = logger
	return o, nil
}

// walkConfigFromCfg projects the pipeline config section to a WalkConfig.
// EntrySelector is left nil to keep the IsXML default; HUPD overrides it.
func walkConfigFromCfg() pipeline.WalkConfig {
	return pipeline.WalkConfig{
		SpoolDir:      cfg.Pipeline.SpoolDir,
		KeepExtracted: cfg.Pipeline.KeepExtracted,
		ExtractedDir:  cfg.Pipeline.ExtractedDir,
	}
}

func localJobsFromDir(dir string) ([]pipeline.ArchiveJob, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	// Drop directories and unknown archive kinds, then project to ArchiveJob.
	return F.Pipe2(
		entries,
		array.Filter(func(e os.DirEntry) bool {
			return !e.IsDir() && pipeline.DetectKind(e.Name()) != pipeline.KindUnknown
		}),
		array.Map(func(e os.DirEntry) pipeline.ArchiveJob {
			return pipeline.ArchiveJob{Name: e.Name(), LocalPath: filepath.Join(dir, e.Name())}
		}),
	), nil
}

// buildCheckpointer returns a Checkpointer plus a closer suitable for
// defer. An empty path yields NoopCheckpointer + a no-op closer.
func buildCheckpointer(path string) (pipeline.Checkpointer, func() error, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return pipeline.NoopCheckpointer{}, func() error { return nil }, nil
	}
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o750); err != nil {
			return nil, nil, fmt.Errorf("checkpoint dir: %w", err)
		}
	}
	cp, err := pipeline.OpenBoltCheckpointer(path)
	if err != nil {
		return nil, nil, err
	}
	return cp, cp.Close, nil
}

// attachCheckpointJanitor wires the CheckpointJanitor into the concrete
// opener type.
func attachCheckpointJanitor(opener pipeline.ArchiveOpener, cp pipeline.Checkpointer) {
	j := pipeline.CheckpointJanitor{CP: cp, Logger: logger}
	switch o := opener.(type) {
	case *pipeline.HTTPOpener:
		o.Janitor = j
	case *pipeline.LocalFileOpener:
		o.Janitor = j
	}
}

// resolveOutputPath returns the configured path for fresh runs and a
// timestamped sibling shard for resumed runs (Parquet has no in-place
// append). Shard form: <stem>.part-<unix>.parquet.
func resolveOutputPath(configured string, resume bool) string {
	if !resume || configured == "" {
		return configured
	}
	dir := filepath.Dir(configured)
	base := filepath.Base(configured)
	ext := filepath.Ext(base)
	stem := strings.TrimSuffix(base, ext)
	if ext == "" {
		ext = ".parquet"
	}
	return filepath.Join(dir, fmt.Sprintf("%s.part-%d%s", stem, time.Now().Unix(), ext))
}

// resetState removes the checkpoint database, the configured Parquet
// output, and any sibling .part-*.parquet shards. Missing files are
// silently ignored.
func resetState(cpPath, outParquet string) error {
	if cpPath = strings.TrimSpace(cpPath); cpPath != "" {
		if err := os.Remove(cpPath); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("remove checkpoint %s: %w", cpPath, err)
		}
		logger.Infow("reset: checkpoint removed", "path", cpPath)
	}
	if outParquet = strings.TrimSpace(outParquet); outParquet != "" {
		if err := os.Remove(outParquet); err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("remove parquet %s: %w", outParquet, err)
		}
		dir := filepath.Dir(outParquet)
		base := filepath.Base(outParquet)
		ext := filepath.Ext(base)
		stem := strings.TrimSuffix(base, ext)
		if ext == "" {
			ext = ".parquet"
		}
		shards, _ := filepath.Glob(filepath.Join(dir, fmt.Sprintf("%s.part-*%s", stem, ext)))
		for _, s := range shards {
			if err := os.Remove(s); err != nil && !os.IsNotExist(err) {
				return fmt.Errorf("remove shard %s: %w", s, err)
			}
		}
		logger.Infow("reset: parquet output removed", "path", outParquet, "shards", len(shards))
	}
	return nil
}

// fmtBytes formats downloaded/total as e.g. "1.2 GB / 4.5 GB" or
// "1.2 GB" when total is unknown (-1) or zero. Returns "" when both
// are zero (no archive in flight).
func fmtBytes(done, total int64) string {
	if done == 0 && total <= 0 {
		return ""
	}
	if total <= 0 {
		return humanBytes(done)
	}
	return humanBytes(done) + " / " + humanBytes(total)
}

func humanBytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for v := n / unit; v >= unit; v /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(n)/float64(div), "KMGTPE"[exp])
}

// mergeParquetShards merges all <stem>.part-*<ext> shard files (plus the
// main outPath if it exists) into a single outPath, then deletes the shards.
// No-op when no shards are found.
func mergeParquetShards(outPath string, log *zap.SugaredLogger) error {
	dir := filepath.Dir(outPath)
	base := filepath.Base(outPath)
	ext := filepath.Ext(base)
	stem := strings.TrimSuffix(base, ext)
	if ext == "" {
		ext = ".parquet"
	}
	shards, _ := filepath.Glob(filepath.Join(dir, stem+".part-*"+ext))
	if len(shards) == 0 {
		return nil
	}

	// Collect sources: existing main file (if present) then shards.
	sources := array.Filter(func(p string) bool {
		_, err := os.Stat(p)
		return err == nil
	})(append([]string{outPath}, shards...))

	log.Infow("merge: merging parquet shards",
		"main", outPath, "shards", len(shards), "total_files", len(sources))

	tmpPath := outPath + ".merging"
	err := withParquetWriter(tmpPath, func(w *parquet.GenericWriter[pipeline.PatentRecord]) error {
		return F.Pipe1(sources,
			array.Reduce(func(acc error, src string) error {
				if acc != nil {
					return acc
				}
				return withParquetReader(src, func(r *parquet.GenericReader[pipeline.PatentRecord]) error {
					return copyParquetRecords(w, r)
				})
			}, error(nil)),
		)
	})
	if err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("merge: %w", err)
	}

	if err := os.Rename(tmpPath, outPath); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("merge rename: %w", err)
	}
	for _, s := range shards {
		if err := os.Remove(s); err != nil && !os.IsNotExist(err) {
			log.Warnw("merge: could not delete shard", "shard", s, "err", err)
		}
	}
	log.Infow("merge: done", "output", outPath, "shards_deleted", len(shards))
	return nil
}

// withParquetWriter opens a GenericWriter on path, calls fn, then closes and
// removes the file on fn error.
func withParquetWriter(path string, fn func(*parquet.GenericWriter[pipeline.PatentRecord]) error) error {
	f, err := os.Create(path) //nolint:gosec // path is config-controlled
	if err != nil {
		return err
	}
	w := parquet.NewGenericWriter[pipeline.PatentRecord](f)
	fnErr := fn(w)
	if fnErr != nil {
		_ = w.Close()
		_ = f.Close()
		return fnErr
	}
	if err := w.Close(); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

// withParquetReader opens a GenericReader on path and calls fn, closing on exit.
func withParquetReader(path string, fn func(*parquet.GenericReader[pipeline.PatentRecord]) error) error {
	f, err := os.Open(path) //nolint:gosec // path is config-controlled
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	r := parquet.NewGenericReader[pipeline.PatentRecord](f)
	defer func() { _ = r.Close() }()
	return fn(r)
}

// copyParquetRecords streams all rows from r into w in fixed-size batches.
func copyParquetRecords(w *parquet.GenericWriter[pipeline.PatentRecord], r *parquet.GenericReader[pipeline.PatentRecord]) error {
	batch := make([]pipeline.PatentRecord, 4096)
	for {
		n, err := r.Read(batch)
		if n > 0 {
			if _, werr := w.Write(batch[:n]); werr != nil {
				return werr
			}
		}
		if err != nil {
			return nil // io.EOF is normal; other errors are swallowed (partial read)
		}
	}
}
