package extract

import (
	"archive/zip"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"time"

	"github.com/IBM/fp-go/v2/function"
	IOE "github.com/IBM/fp-go/v2/ioeither"
	"github.com/schollz/progressbar/v3"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"

	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/config"
	T "github.com/Qubut/IP-Claim/packages/epo_processor/internal/typing"
)

type Extractor struct {
	Cfg             config.Config
	DeleteAfter     bool
	progress        *progressbar.ProgressBar
	ExtractedFiles  *atomic.Int64
	currentArchive  string
	currentFile     string
	Logger          *zap.SugaredLogger
	Tracer          trace.Tracer
	Meter           metric.Meter
	sessionDuration metric.Int64Histogram
	filesTotal      metric.Int64Counter
	zipsTotal       metric.Int64Counter
	zipsFailed      metric.Int64Counter
	bytesTotal      metric.Int64Counter
	fileDuration    metric.Int64Histogram
}

func NewExtractor(
	cfg config.Config,
	tracer trace.Tracer,
	logger *zap.SugaredLogger,
	meter metric.Meter,
) (*Extractor, error) {
	e := &Extractor{
		DeleteAfter:    cfg.Extract.DeleteAfterExtract,
		ExtractedFiles: &atomic.Int64{},
		Logger:         logger,
		Tracer:         tracer,
		Meter:          meter,
		Cfg:            cfg,
	}

	var err error

	e.sessionDuration, err = meter.Int64Histogram(
		"extraction.session.duration",
		metric.WithDescription("Duration of the full extraction session"),
		metric.WithUnit("ms"),
	)
	if err != nil {
		return nil, err
	}

	e.filesTotal, err = meter.Int64Counter(
		"extraction.files.total",
		metric.WithDescription("Total number of files extracted"),
	)
	if err != nil {
		return nil, err
	}

	e.zipsTotal, err = meter.Int64Counter(
		"extraction.zips.total",
		metric.WithDescription("Number of zip archives processed"),
	)
	if err != nil {
		return nil, err
	}

	e.zipsFailed, err = meter.Int64Counter(
		"extraction.zips.failed",
		metric.WithDescription("Number of zip archives that failed to extract"),
	)
	if err != nil {
		return nil, err
	}

	e.bytesTotal, err = meter.Int64Counter(
		"extraction.bytes.total",
		metric.WithDescription("Total bytes extracted"),
		metric.WithUnit("By"),
	)
	if err != nil {
		return nil, err
	}

	e.fileDuration, err = meter.Int64Histogram(
		"extraction.file.duration",
		metric.WithDescription("Duration of individual file extraction"),
		metric.WithUnit("ms"),
	)
	if err != nil {
		return nil, err
	}

	return e, nil
}

func (e *Extractor) ExtractAll(ctx context.Context, dir string) IOE.IOEither[error, T.Unit] {
	ctx, span := e.Tracer.Start(ctx, "extraction.session", trace.WithAttributes(
		attribute.String("directory", dir),
		attribute.Bool("delete_after", e.DeleteAfter),
	))
	defer span.End()
	startTime := time.Now()
	e.Logger.Infow("Starting extraction in directory", "dir", dir, "deleteAfter", e.DeleteAfter)

	e.progress = progressbar.NewOptions64(-1,
		progressbar.OptionSetWriter(os.Stdout),
		progressbar.OptionSetWidth(60),
		progressbar.OptionSetDescription("[0 extracted] Finding zip files..."),
		progressbar.OptionSpinnerType(14),
		progressbar.OptionSetElapsedTime(true),
		progressbar.OptionSetPredictTime(true),
		progressbar.OptionThrottle(50*time.Millisecond),
		progressbar.OptionSetRenderBlankState(true),
		progressbar.OptionUseANSICodes(true),
	)

	select {
	case <-ctx.Done():
		e.Logger.Warn("Extraction session cancelled")
		return IOE.Left[T.Unit](ctx.Err())
	default:
	}
	return function.Pipe2(
		IOE.TryCatchError(func() ([]string, error) {
			return e.findZipFiles(dir)
		}),
		IOE.Chain(func(zipFiles []string) IOE.IOEither[error, []T.Unit] {
			select {
			case <-ctx.Done():
				return IOE.Left[[]T.Unit](ctx.Err())
			default:
			}
			e.zipsTotal.Add(ctx, int64(len(zipFiles)),
				metric.WithAttributes(
					attribute.String("type", "main"),
				),
			)
			if len(zipFiles) == 0 {
				e.Logger.Infow("No zip files found in directory", "dir", dir)
				return IOE.Right[error]([]T.Unit{})
			}

			e.Logger.Infow("Found zip files to extract", "count", len(zipFiles), "dir", dir)
			e.progress.Describe(
				fmt.Sprintf("[0 extracted] Processing %d zip files...", len(zipFiles)),
			)

			traverse := IOE.TraverseArrayPar(func(zipPath string) IOE.IOEither[error, T.Unit] {
				select {
				case <-ctx.Done():
					return IOE.Left[T.Unit](ctx.Err())
				default:
					return e.processSingleZip(ctx, zipPath)
				}
			})
			return traverse(zipFiles)
		}),
		IOE.Map[error](func(_ []T.Unit) T.Unit {
			durationMs := time.Since(startTime).Milliseconds()
			status := "success"
			if e.ExtractedFiles.Load() == 0 {
				status = "empty"
			}

			e.sessionDuration.Record(ctx, durationMs,
				metric.WithAttributes(
					attribute.String("status", status),
					attribute.Bool("delete_after", e.DeleteAfter),
				),
			)

			if e.progress != nil {
				e.progress.Describe("Extraction complete")
				_ = e.progress.Finish()
				e.progress = nil
				e.Logger.Infow("Extraction completed", "total_files", e.ExtractedFiles.Load())
			}
			return T.Unit{}
		}),
	)
}

func (e *Extractor) ProcessZipFile(zipPath string) IOE.IOEither[error, T.Unit] {
	ctx := context.Background()
	return e.processSingleZip(ctx, zipPath)
}

func (e *Extractor) processSingleZip(
	ctx context.Context,
	zipPath string,
) IOE.IOEither[error, T.Unit] {
	ctx, span := e.Tracer.Start(ctx, "process.zip", trace.WithAttributes(
		attribute.String("zip_path", zipPath),
	))
	defer span.End()
	startTime := time.Now()
	baseName := strings.TrimSuffix(filepath.Base(zipPath), ".zip")
	destDir := filepath.Join(filepath.Dir(zipPath), baseName)
	e.Logger.Infow("Processing zip file",
		"zip", zipPath,
		"baseName", baseName,
		"destDir", destDir,
	)
	select {
	case <-ctx.Done():
		return IOE.Left[T.Unit](ctx.Err())
	default:
	}
	return function.Pipe3(
		IOE.TryCatchError(func() (T.Unit, error) {
			select {
			case <-ctx.Done():
				return T.Unit{}, ctx.Err()
			default:
			}
			e.Logger.Infow("Extracting main archive", "zip", zipPath, "dest", destDir)
			e.currentArchive = zipPath
			e.progress.Describe(fmt.Sprintf("Extracting %s", filepath.Base(zipPath)))
			return T.Unit{}, e.extractZipToDir(zipPath, destDir)
		}),
		IOE.Chain(func(_ T.Unit) IOE.IOEither[error, T.Unit] {
			select {
			case <-ctx.Done():
				return IOE.Left[T.Unit](ctx.Err())
			default:
			}
			e.progress.Describe(fmt.Sprintf("Extracting nested zips in %s", baseName))
			return e.extractAllZipsInDir(ctx, destDir)
		}),
		IOE.Chain(func(_ T.Unit) IOE.IOEither[error, T.Unit] {
			select {
			case <-ctx.Done():
				return IOE.Left[T.Unit](ctx.Err())
			default:
			}
			if e.DeleteAfter {
				return IOE.TryCatchError(func() (T.Unit, error) {
					if err := os.Remove(zipPath); err != nil {
						e.Logger.Warnw(
							"Failed to delete original zip",
							"zip",
							zipPath,
							"error",
							err,
						)
					} else {
						e.Logger.Infow("Deleted original zip", "zip", zipPath)
					}
					return T.Unit{}, nil
				})
			}
			return IOE.Right[error](T.Unit{})
		}),
		IOE.Tap(func(_ T.Unit) IOE.IOEither[error, T.Unit] {
			durationMs := time.Since(startTime).Milliseconds()
			e.fileDuration.Record(ctx, durationMs,
				metric.WithAttributes(
					attribute.String("status", "success"),
					attribute.String("type", "zip"),
				),
			)
			return IOE.Of[error](T.Unit{})
		}),
	)
}

func (e *Extractor) findZipFiles(dir string) ([]string, error) {
	var zipFiles []string

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	for _, entry := range entries {
		if !entry.IsDir() && strings.HasSuffix(strings.ToLower(entry.Name()), ".zip") {
			zipFiles = append(zipFiles, filepath.Join(dir, entry.Name()))
		}
	}

	return zipFiles, nil
}

func (e *Extractor) updateDescription() {
	if e.progress != nil {
		desc := fmt.Sprintf("[%d extracted] Extracting %s", e.ExtractedFiles.Load(), e.currentFile)
		if e.currentArchive != "" {
			desc = fmt.Sprintf("[%d extracted] Extracting %s from %s",
				e.ExtractedFiles.Load(), e.currentFile, filepath.Base(e.currentArchive))
		}
		e.progress.Describe(desc)
	}
}

func (e *Extractor) extractAllZipsInDir(
	ctx context.Context,
	dir string,
) IOE.IOEither[error, T.Unit] {
	return IOE.TryCatchError(func() (T.Unit, error) {
		for {
			select {
			case <-ctx.Done():
				e.Logger.Warn("Nested zip extraction cancelled")
				return T.Unit{}, ctx.Err()
			default:
			}
			zipFiles, err := e.findAllZipFilesRecursive(dir)
			if err != nil {
				return T.Unit{}, err
			}
			if len(zipFiles) == 0 {
				break
			}

			e.Logger.Debugw("Found nested zip files", "count", len(zipFiles), "dir", dir)

			for _, zipFile := range zipFiles {
				select {
				case <-ctx.Done():
					return T.Unit{}, ctx.Err()
				default:
				}
				ctx, span := e.Tracer.Start(ctx, "extract.nested_zip", trace.WithAttributes(
					attribute.String("zip_file", zipFile),
				))

				e.Logger.Infow("Extracting nested zip", "zip", zipFile)

				e.zipsTotal.Add(ctx, 1,
					metric.WithAttributes(
						attribute.String("type", "nested"),
					),
				)

				destDir := filepath.Dir(zipFile)
				if err := e.extractZipToDir(zipFile, destDir); err != nil {
					span.RecordError(err)
					span.End()
					e.zipsFailed.Add(ctx, 1,
						metric.WithAttributes(
							attribute.String("error_type", "extract_failed"),
						),
					)
					return T.Unit{}, err
				}

				if e.DeleteAfter {
					if err := os.Remove(zipFile); err != nil {
						e.Logger.Warnw("Failed to delete zip file", "zip", zipFile, "error", err)
					} else {
						e.Logger.Infow("Deleted zip file", "zip", zipFile)
					}
				}

				span.End()
			}
		}
		return T.Unit{}, nil
	})
}

func (e *Extractor) findAllZipFilesRecursive(dir string) ([]string, error) {
	var zipFiles []string

	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if !d.IsDir() && strings.HasSuffix(strings.ToLower(d.Name()), ".zip") {
			zipFiles = append(zipFiles, path)
		}

		return nil
	})

	return zipFiles, err
}

func (e *Extractor) extractZipToDir(zipPath, destDir string) error {
	startTime := time.Now()
	e.Logger.Debugw("Opening zip file", "zip", zipPath, "dest", destDir)

	r, err := zip.OpenReader(zipPath)
	if err != nil {
		return fmt.Errorf("failed to open zip %s: %w", zipPath, err)
	}
	defer r.Close()

	e.currentArchive = zipPath
	e.Logger.Debugw("Zip opened", "file_count", len(r.File), "zip", zipPath)

	for _, f := range r.File {
		e.currentFile = f.Name
		e.updateDescription()

		destPath := filepath.Join(destDir, f.Name)

		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(destPath, os.ModePerm); err != nil {
				return fmt.Errorf("failed to create directory %s: %w", destPath, err)
			}
			continue
		}

		if err := os.MkdirAll(filepath.Dir(destPath), os.ModePerm); err != nil {
			return fmt.Errorf("failed to create parent directory for %s: %w", destPath, err)
		}

		rc, err := f.Open()
		if err != nil {
			return fmt.Errorf("failed to open file %s in zip: %w", f.Name, err)
		}

		destFile, err := os.Create(destPath)
		if err != nil {
			rc.Close()
			return fmt.Errorf("failed to create file %s: %w", destPath, err)
		}

		n, err := io.Copy(destFile, rc)

		destFile.Close()
		rc.Close()

		if err != nil {
			return fmt.Errorf("failed to copy file %s: %w", f.Name, err)
		}

		durationMs := time.Since(startTime).Milliseconds()
		e.fileDuration.Record(context.Background(), durationMs,
			metric.WithAttributes(
				attribute.String("status", "success"),
				attribute.Bool("nested", false),
			),
		)

		e.filesTotal.Add(context.Background(), 1)
		e.bytesTotal.Add(context.Background(), n)
		e.ExtractedFiles.Add(1)
		e.updateDescription()

		e.Logger.Debugw("File extracted", "file", f.Name, "dest", destPath)
	}

	e.Logger.Infow("Zip extraction completed", "zip", zipPath, "files_extracted", len(r.File))
	return nil
}
