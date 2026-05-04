package cmd

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
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

var processHupdCmd = &cobra.Command{
	Use:   "process-hupd",
	Short: "Stream the HUPD .tar to disk (archive + unwrapped contents), no parsing",
		RunE: func(_ *cobra.Command, _ []string) error {
		ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer cancel()

		hupd := cfg.HUPD
		if strings.TrimSpace(hupd.URL) == "" {
			return fmt.Errorf("download.hupd.url is empty")
		}
		if !cfg.Pipeline.KeepArchive && !cfg.Pipeline.KeepExtracted {
			return fmt.Errorf(
				"process-hupd is a no-op unless pipeline.keep_archive or pipeline.keep_extracted is true",
			)
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

		bar := newProgressBar("[cyan]downloading HUPD[reset]")
		var (
			barMu      sync.Mutex
			lastLogged time.Time
		)
		opener.OnBytes = func(job pipeline.ArchiveJob, downloaded, total int64) {
			barMu.Lock()
			defer barMu.Unlock()
			bar.Describe(fmt.Sprintf("[cyan]HUPD[reset] %s  [white]%s[reset]",
				job.Name, fmtBytes(downloaded, total)))
			_ = bar.Set64(downloaded)
			if now := time.Now(); now.Sub(lastLogged) >= 5*time.Second {
				lastLogged = now
				logger.Infow("download: progress", "label", job.Name, "bytes", fmtBytes(downloaded, total))
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
			pipeline.WithExtractorConcurrency(cfg.Pipeline.ExtractorConcurrency),
			pipeline.WithBatchSize(1),
			pipeline.WithBatchTimeout(cfg.Pipeline.BatchTimeout),
		)
		if err != nil {
			return fmt.Errorf("build pipeline: %w", err)
		}

		logger.Infow("Streaming HUPD",
			"url", hupd.URL,
			"keep_archive", cfg.Pipeline.KeepArchive,
			"archive_dir", cfg.Pipeline.ArchiveDir,
			"keep_extracted", cfg.Pipeline.KeepExtracted,
			"extracted_dir", cfg.Pipeline.ExtractedDir,
		)
		if err := p.Run(ctx); err != nil {
			_ = bar.Finish()
			return fmt.Errorf("hupd pipeline: %w", err)
		}
		_ = bar.Finish()
		logger.Info("HUPD streaming complete")
		return nil
	},
}
