package cmd

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/pipeline"
)

// processHupdCmd streams the HUPD HuggingFace .tar to disk without
// parsing. Use it to materialise the raw archive plus its unwrapped
// contents for downstream tooling.
//
// hupdMaxRetries returns a per-archive retry budget, falling back to 3
// when Server.MaxRetries is unset.
func hupdMaxRetries(serverMaxRetries int) uint {
	if serverMaxRetries <= 0 {
		return 3
	}
	return uint(serverMaxRetries)
}

func init() {
	flagStr(processHupdCmd, "hupd.url", "url", "u", "", "HUPD .tar URL")
	flagStr(processHupdCmd, "hupd.filename", "filename", "", "", "Archive name used in logs/checkpoint")
	flagBool(processHupdCmd, "pipeline.keep_archive", "keep-archive", "", false, "Also keep the raw .tar under --archive-dir")
	flagStr(processHupdCmd, "pipeline.archive_dir", "archive-dir", "", "", "Where to keep the raw .tar (with --keep-archive)")
	flagStr(processHupdCmd, "pipeline.spool_dir", "spool-dir", "", "", "Zip spool dir (defaults to OS temp)")
	flagInt(processHupdCmd, "pipeline.extractor_concurrency", "extract-concurrency", "", 4, "Parallel extractors")
	flagInt(processHupdCmd, "server.max_retries", "max-retries", "", 3, "Max HTTP retries")
}

var processHupdCmd = &cobra.Command{
	Use:   "process-hupd <out-dir>",
	Short: "Stream the HUPD .tar to disk (archive + unwrapped contents), no parsing",
	Long: "Stream the HUPD all-years .tar from HuggingFace and write its unwrapped\n" +
		"entries under <out-dir>. Pass --keep-archive to also retain the raw .tar.",
	Args: cobra.ExactArgs(1),
	RunE: func(_ *cobra.Command, args []string) error {
		ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer cancel()

		// The positional <out-dir> is the extraction target; selecting it
		// implies retention, so process-hupd is never a no-op.
		cfg.Pipeline.ExtractedDir = args[0]
		cfg.Pipeline.KeepExtracted = true

		hupd := cfg.HUPD
		if strings.TrimSpace(hupd.URL) == "" {
			return fmt.Errorf("HUPD URL is empty: pass --url or set hupd.url in config")
		}

		walk := pipeline.WalkConfig{
			SpoolDir:      cfg.Pipeline.SpoolDir,
			KeepExtracted: cfg.Pipeline.KeepExtracted,
			ExtractedDir:  cfg.Pipeline.ExtractedDir,
			// HUPD payload entries are JSON, not XML; emit them. Nested
			// .tar / .tar.gz / .zip entries (HUPD ships per-year shards
			// inside all-years.tar) must fall through to the recursion
			// branch in dispatchEntry, so the selector skips archive kinds.
			EntrySelector: func(name string) bool {
				return pipeline.DetectKind(name) == pipeline.KindUnknown
			},
		}
		opener := pipeline.NewHTTPOpener(
			&http.Client{Timeout: 0}, // long streamed download
			hupdMaxRetries(cfg.Server.MaxRetries),
			false, // HUPD has no published per-file SHA-1 manifest
			walk,
		)
		opener.KeepArchive = cfg.Pipeline.KeepArchive
		opener.ArchiveDir = cfg.Pipeline.ArchiveDir
		opener.Logger = logger

		disp := newDisplay(ctx)
		tty := disp.active()
		if tty {
			logCtl.Console.SetTTYMode(disp.writer())
			defer logCtl.Console.Restore()
		}
		bar := disp.singleBar("downloading HUPD")
		defer disp.stop()

		lastStats := newRef(pipeline.Stats{})
		lastLogged := newRef(time.Time{})
		opener.OnBytes = func(job pipeline.ArchiveJob, downloaded, total int64) {
			bar.Describe(fmt.Sprintf("HUPD %s  %s", job.Name, fmtBytes(downloaded, total)))
			if now := time.Now(); now.Sub(lastLogged.Get()()) >= 5*time.Second {
				lastLogged.Set(now)()
				logger.Info("download: progress", "label", job.Name, "bytes", fmtBytes(downloaded, total))
			}
		}

		src := &pipeline.StaticListSource{Jobs: []pipeline.ArchiveJob{{
			Name: hupd.Filename,
			URL:  hupd.URL,
		}}}

		p, err := pipeline.New(
			pipeline.WithSource(src),
			pipeline.WithOpener(opener),
			pipeline.WithExtractor(pipeline.NoopExtractor{}),
			pipeline.WithSink(pipeline.NoopSink{}),
			pipeline.WithLogger(logger),
			pipeline.WithArchiveConcurrency(1), // single archive
			pipeline.WithExtractorConcurrency(resolveReaderConcurrency()),
			pipeline.WithBatchSize(1),
			pipeline.WithBatchTimeout(cfg.Pipeline.BatchTimeout),
			pipeline.WithProgress(func(s pipeline.Stats) { lastStats.Set(s)() }),
		)
		if err != nil {
			return fmt.Errorf("build pipeline: %w", err)
		}

		logger.Info("Streaming HUPD",
			"url", hupd.URL,
			"keep_archive", cfg.Pipeline.KeepArchive,
			"archive_dir", cfg.Pipeline.ArchiveDir,
			"keep_extracted", cfg.Pipeline.KeepExtracted,
			"extracted_dir", cfg.Pipeline.ExtractedDir,
		)
		started := time.Now()
		runErr := p.Run(ctx)
		disp.stop()
		if tty {
			logCtl.Console.Restore()
		}
		if runErr != nil {
			return fmt.Errorf("hupd pipeline: %w", runErr)
		}
		elapsed := time.Since(started)
		logger.Info("HUPD streaming complete")

		final := lastStats.Get()()
		renderSummary("HUPD process — summary", []kv{
			{"Archives", strconv.FormatInt(final.Archives, 10)},
			{"Entries", strconv.FormatInt(final.Entries, 10)},
			{"Archive dir", cfg.Pipeline.ArchiveDir},
			{"Extracted dir", cfg.Pipeline.ExtractedDir},
			{"Elapsed", elapsed.Round(time.Millisecond).String()},
			{"Entries/s", fmt.Sprintf("%.1f", ratePerSec(final.Entries, elapsed))},
		})
		renderIssues(logCtl)
		return nil
	},
}
