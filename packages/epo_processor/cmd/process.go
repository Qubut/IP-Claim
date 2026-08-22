package cmd

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/IBM/fp-go/v2/array"
	F "github.com/IBM/fp-go/v2/function"
	IOR "github.com/IBM/fp-go/v2/idiomatic/ioresult"
	"github.com/parquet-go/parquet-go"
	"github.com/spf13/cobra"

	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/pipeline"
)

func init() {
	// Pipeline output / tuning.
	flagStr(processCmd, "pipeline.output_parquet", "out", "o", "./data.parquet", "Parquet output path")
	flagInt(processCmd, "pipeline.archive_concurrency", "concurrency", "c", 4, "Parallel archive downloads/walks")
	flagInt(processCmd, "pipeline.extractor_concurrency", "extract-concurrency", "", 4, "Archive entries decoded in parallel")
	flagInt(processCmd, "pipeline.parser_concurrency", "parser-concurrency", "", 0, "Per-document parse workers per entry (0 = NumCPU)")
	flagInt(processCmd, "pipeline.batch_size", "batch-size", "", 1000, "Rows per Parquet write")
	flagDur(processCmd, "pipeline.batch_timeout", "batch-timeout", "", 2*time.Second, "Idle flush timeout")
	flagInt(processCmd, "pipeline.row_group_size", "row-group-size", "", 50000, "Records per Parquet row group (caps RAM)")
	flagStr(processCmd, "pipeline.spool_dir", "spool-dir", "", "", "Zip spool dir (defaults to OS temp)")
	flagStr(processCmd, "pipeline.use_local_dir", "local-dir", "", "", "Replay archives from this dir, no network")
	// Retention.
	flagBool(processCmd, "pipeline.keep_archive", "keep-archive", "", false, "Tee HTTP body to --archive-dir")
	flagStr(processCmd, "pipeline.archive_dir", "archive-dir", "", "", "Where to keep raw archives")
	flagBool(processCmd, "pipeline.keep_extracted", "keep-extracted", "", false, "Tee unwrapped entries to --extracted-dir")
	flagStr(processCmd, "pipeline.extracted_dir", "extracted-dir", "", "", "Where to keep unwrapped entries")
	// Resumability.
	flagStr(processCmd, "pipeline.checkpoint_db", "checkpoint", "", "", "bbolt path enabling resumable runs (empty disables)")
	flagBool(processCmd, "pipeline.reset_checkpoint", "reset", "", false, "Delete checkpoint DB and existing parquet output(s) before running")
	// Server (live EPO source).
	flagStr(processCmd, "server.base_url", "base-url", "", "", "EPO BDDS base URL")
	flagDur(processCmd, "server.timeout", "timeout", "", 30*time.Second, "Per-request HTTP timeout")
	flagInt(processCmd, "server.max_retries", "max-retries", "", 3, "Max HTTP retries per archive")
	flagInt(processCmd, "server.product_id", "product-id", "", 3, "EPO product ID")
	flagBool(processCmd, "server.verify_sha1", "verify-sha1", "", false, "Verify per-archive SHA-1")
}

// newConfiguredExtractor builds the EPO XML extractor with the parse
// concurrency taken from config.
func newConfiguredExtractor() *pipeline.XMLStreamExtractor {
	x := pipeline.NewXMLStreamExtractor()
	x.ParseConcurrency = cfg.Pipeline.ParserConcurrency
	return x
}

// resolveReaderConcurrency turns the configured (possibly auto) extractor
// concurrency into a concrete reader count, auto-sized to NumCPU and clamped
// by the configured memory budget.
func resolveReaderConcurrency() int {
	return pipeline.PlanReaderConcurrency(
		cfg.Pipeline.ExtractorConcurrency,
		runtime.NumCPU(),
		int64(cfg.Pipeline.MemoryBudgetGB)<<30,
		int64(cfg.Pipeline.PerEntryEstimateMB)<<20,
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
			logger.Warn("resume: writing to a new shard to preserve previous output",
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
			logger.Info("checkpoint: enabled", "db", cfg.Pipeline.CheckpointDB, "resume", resume)
		}

		// Live multi-bar display (TTY only). Route console logs through it so
		// chatty INFO is suppressed and WARN+ prints cleanly above the bars.
		disp := newDisplay(ctx)
		tty := disp.active()
		if tty {
			logCtl.Console.SetTTYMode(disp.writer())
			defer logCtl.Console.Restore() // safety net for early returns
		}
		disp.startAggregate("streaming EPO")
		defer disp.stop()

		readerConc := resolveReaderConcurrency()
		logger.Info("reader concurrency resolved",
			"requested", cfg.Pipeline.ExtractorConcurrency,
			"num_cpu", runtime.NumCPU(),
			"memory_budget_gb", cfg.Pipeline.MemoryBudgetGB,
			"per_entry_estimate_mb", cfg.Pipeline.PerEntryEstimateMB,
			"resolved", readerConc,
		)

		// lastStats is shared between the stats goroutine (writer) and this
		// goroutine (final reader). The ref cell guards both accesses.
		lastStats := newRef(pipeline.Stats{})
		opts := []pipeline.Option{
			pipeline.WithSource(source),
			pipeline.WithOpener(opener),
			pipeline.WithExtractor(newConfiguredExtractor()),
			pipeline.WithSink(sink),
			pipeline.WithLogger(logger),
			pipeline.WithArchiveConcurrency(cfg.Pipeline.ArchiveConcurrency),
			pipeline.WithExtractorConcurrency(readerConc),
			pipeline.WithBatchSize(cfg.Pipeline.BatchSize),
			pipeline.WithBatchTimeout(cfg.Pipeline.BatchTimeout),
		}
		opts = append(opts, pipeline.WithProgress(func(s pipeline.Stats) {
			lastStats.Set(s)()
			disp.setAggregate(fmt.Sprintf(
				"archives %d  entries %d  records %d  batches %d",
				s.Archives, s.Entries, s.Records, s.Batches,
			))
		}))
		if h, ok := opener.(*pipeline.HTTPOpener); ok {
			h.OnBytes = func(job pipeline.ArchiveJob, downloaded, total int64) {
				disp.onBytes(job, downloaded, total)
			}
			h.OnArchiveSettled = func(job pipeline.ArchiveJob, err error) {
				disp.onSettled(job, err)
			}
		}

		p, err := pipeline.New(opts...)
		if err != nil {
			return fmt.Errorf("build pipeline: %w", err)
		}

		logger.Info("Starting streaming pipeline",
			"output", outPath,
			"archive_concurrency", cfg.Pipeline.ArchiveConcurrency,
			"extractor_concurrency", readerConc,
			"batch_size", cfg.Pipeline.BatchSize,
		)
		started := time.Now()
		runErr := p.Run(ctx)
		// Tear the bars down and hand the terminal back to plain logging
		// before printing anything else, so neither the post-run logs nor the
		// summary table are clobbered (and we stop writing through the now-shut
		// mpb container).
		disp.stop()
		if tty {
			logCtl.Console.Restore()
		}
		if runErr != nil {
			return fmt.Errorf("pipeline: %w", runErr)
		}
		elapsed := time.Since(started)
		logger.Info("Streaming pipeline completed")

		// Merge shards into the main output file so the workspace stays tidy
		// and downstream tools (analyze) always read a single file.
		if err := mergeParquetShards(cfg.Pipeline.OutputParquet, logger); err != nil {
			logger.Warn("merge shards: failed (non-fatal)", "err", err)
		}

		// Read the last published stats (zero value when no progress fired).
		final := lastStats.Get()()
		renderSummary("EPO process — summary", []kv{
			{"Archives", strconv.FormatInt(final.Archives, 10)},
			{"Entries", strconv.FormatInt(final.Entries, 10)},
			{"Records", strconv.FormatInt(final.Records, 10)},
			{"Batches", strconv.FormatInt(final.Batches, 10)},
			{"Output", cfg.Pipeline.OutputParquet},
			{"Elapsed", elapsed.Round(time.Millisecond).String()},
			{"Records/s", fmt.Sprintf("%.1f", ratePerSec(final.Records, elapsed))},
			{"Entries/s", fmt.Sprintf("%.1f", ratePerSec(final.Entries, elapsed))},
		})
		renderIssues(logCtl)
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
		if err := removeIfExists(cpPath); err != nil {
			return fmt.Errorf("remove checkpoint %s: %w", cpPath, err)
		}
		logger.Info("reset: checkpoint removed", "path", cpPath)
	}
	if outParquet = strings.TrimSpace(outParquet); outParquet != "" {
		if err := removeIfExists(outParquet); err != nil {
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
		if _, err := F.Pipe1(
			shards,
			IOR.TraverseArray(func(s string) IOR.IOResult[any] {
				return func() (any, error) { return nil, removeIfExists(s) }
			}),
		)(); err != nil {
			return fmt.Errorf("remove shard: %w", err)
		}
		logger.Info("reset: parquet output removed", "path", outParquet, "shards", len(shards))
	}
	return nil
}

// removeIfExists deletes path, treating a missing file as success. It is the
// unit effect folded over by the reset traversals.
func removeIfExists(path string) error {
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
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
func mergeParquetShards(outPath string, log *slog.Logger) error {
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

	log.Info("merge: merging parquet shards",
		"main", outPath, "shards", len(shards), "total_files", len(sources))

	tmpPath := outPath + ".merging"
	err := withParquetWriter(tmpPath, func(w *parquet.GenericWriter[pipeline.PatentRecord]) error {
		// TraverseArray sequences each shard read, short-circuiting on the
		// first reader error.
		_, e := F.Pipe1(
			sources,
			IOR.TraverseArray(func(src string) IOR.IOResult[any] {
				return func() (any, error) {
					return nil, withParquetReader(src, func(r *parquet.GenericReader[pipeline.PatentRecord]) error {
						return copyParquetRecords(w, r)
					})
				}
			}),
		)()
		return e
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
			log.Warn("merge: could not delete shard", "shard", s, "err", err)
		}
	}
	log.Info("merge: done", "output", outPath, "shards_deleted", len(shards))
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
