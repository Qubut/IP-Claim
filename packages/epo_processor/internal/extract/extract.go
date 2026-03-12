package extract

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
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

type ArchiveType string

const (
	ZipType     ArchiveType = "zip"
	TarType     ArchiveType = "tar"
	TarGzType   ArchiveType = "tar.gz"
	TgzType     ArchiveType = "tgz"
	UnknownType ArchiveType = "unknown"
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
	archivesTotal   metric.Int64Counter
	archivesFailed  metric.Int64Counter
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

	e.archivesTotal, err = meter.Int64Counter(
		"extraction.archives.total",
		metric.WithDescription("Number of archives processed"),
	)
	if err != nil {
		return nil, err
	}

	e.archivesFailed, err = meter.Int64Counter(
		"extraction.archives.failed",
		metric.WithDescription("Number of archives that failed to extract"),
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
		progressbar.OptionSetDescription("[0 extracted] Finding archive files..."),
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
			return e.findArchiveFiles(dir)
		}),
		IOE.Chain(func(archiveFiles []string) IOE.IOEither[error, []T.Unit] {
			select {
			case <-ctx.Done():
				return IOE.Left[[]T.Unit](ctx.Err())
			default:
			}
			e.archivesTotal.Add(ctx, int64(len(archiveFiles)),
				metric.WithAttributes(
					attribute.String("type", "main"),
				),
			)
			if len(archiveFiles) == 0 {
				e.Logger.Infow("No archive files found in directory", "dir", dir)
				return IOE.Right[error]([]T.Unit{})
			}

			e.Logger.Infow("Found archive files to extract", "count", len(archiveFiles), "dir", dir)
			e.progress.Describe(
				fmt.Sprintf("[0 extracted] Processing %d archive files...", len(archiveFiles)),
			)

			traverse := IOE.TraverseArrayPar(func(archivePath string) IOE.IOEither[error, T.Unit] {
				select {
				case <-ctx.Done():
					return IOE.Left[T.Unit](ctx.Err())
				default:
					return e.processSingleArchive(ctx, archivePath)
				}
			})
			return traverse(archiveFiles)
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

func (e *Extractor) ProcessArchiveFile(archivePath string) IOE.IOEither[error, T.Unit] {
	ctx := context.Background()
	return e.processSingleArchive(ctx, archivePath)
}

func (e *Extractor) processSingleArchive(
	ctx context.Context,
	archivePath string,
) IOE.IOEither[error, T.Unit] {
	archiveType := getArchiveType(archivePath)
	ctx, span := e.Tracer.Start(ctx, "process.archive", trace.WithAttributes(
		attribute.String("archive_path", archivePath),
		attribute.String("archive_type", string(archiveType)),
	))
	defer span.End()
	startTime := time.Now()
	baseName := strings.TrimSuffix(filepath.Base(archivePath), filepath.Ext(archivePath))
	if archiveType == TarGzType || archiveType == TgzType {
		baseName = strings.TrimSuffix(baseName, ".tar") // Remove .tar for .tar.gz/.tgz
	}
	destDir := filepath.Join(filepath.Dir(archivePath), baseName)
	e.Logger.Infow("Processing archive file",
		"archive", archivePath,
		"baseName", baseName,
		"destDir", destDir,
		"archive_type", archiveType,
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
			e.Logger.Infow("Extracting main archive", "archive", archivePath, "dest", destDir)
			e.currentArchive = archivePath
			e.progress.Describe(fmt.Sprintf("Extracting %s", filepath.Base(archivePath)))
			return T.Unit{}, e.extractToDir(archivePath, destDir, archiveType)
		}),
		IOE.Chain(func(_ T.Unit) IOE.IOEither[error, T.Unit] {
			select {
			case <-ctx.Done():
				return IOE.Left[T.Unit](ctx.Err())
			default:
			}
			e.progress.Describe(fmt.Sprintf("Extracting nested archives in %s", baseName))
			return e.extractAllArchivesInDir(ctx, destDir)
		}),
		IOE.Chain(func(_ T.Unit) IOE.IOEither[error, T.Unit] {
			select {
			case <-ctx.Done():
				return IOE.Left[T.Unit](ctx.Err())
			default:
			}
			if e.DeleteAfter {
				return IOE.TryCatchError(func() (T.Unit, error) {
					if err := os.Remove(archivePath); err != nil {
						e.Logger.Warnw(
							"Failed to delete original archive",
							"archive",
							archivePath,
							"error",
							err,
						)
					} else {
						e.Logger.Infow("Deleted original archive", "archive", archivePath)
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
					attribute.String("archive_type", string(archiveType)),
				),
			)
			return IOE.Of[error](T.Unit{})
		}),
	)
}

func getArchiveType(path string) ArchiveType {
	lower := strings.ToLower(path)
	if strings.HasSuffix(lower, ".zip") {
		return ZipType
	} else if strings.HasSuffix(lower, ".tar.gz") {
		return TarGzType
	} else if strings.HasSuffix(lower, ".tgz") {
		return TgzType
	} else if strings.HasSuffix(lower, ".tar") {
		return TarType
	}
	return UnknownType
}

func (e *Extractor) findArchiveFiles(dir string) ([]string, error) {
	var archiveFiles []string

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			if getArchiveType(entry.Name()) != UnknownType {
				archiveFiles = append(archiveFiles, filepath.Join(dir, entry.Name()))
			}
		}
	}

	return archiveFiles, nil
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

func (e *Extractor) extractAllArchivesInDir(
	ctx context.Context,
	dir string,
) IOE.IOEither[error, T.Unit] {
	return IOE.TryCatchError(func() (T.Unit, error) {
		for {
			select {
			case <-ctx.Done():
				e.Logger.Warn("Nested archive extraction cancelled")
				return T.Unit{}, ctx.Err()
			default:
			}
			archiveFiles, err := e.findAllArchiveFilesRecursive(dir)
			if err != nil {
				return T.Unit{}, err
			}
			if len(archiveFiles) == 0 {
				break
			}

			e.Logger.Debugw("Found nested archive files", "count", len(archiveFiles), "dir", dir)

			for _, archiveFile := range archiveFiles {
				select {
				case <-ctx.Done():
					return T.Unit{}, ctx.Err()
				default:
				}
				archiveType := getArchiveType(archiveFile)
				ctx, span := e.Tracer.Start(ctx, "extract.nested_archive", trace.WithAttributes(
					attribute.String("archive_file", archiveFile),
					attribute.String("archive_type", string(archiveType)),
				))

				e.Logger.Infow(
					"Extracting nested archive",
					"archive",
					archiveFile,
					"type",
					archiveType,
				)

				e.archivesTotal.Add(ctx, 1,
					metric.WithAttributes(
						attribute.String("type", "nested"),
						attribute.String("archive_type", string(archiveType)),
					),
				)

				destDir := filepath.Dir(archiveFile)
				if err := e.extractToDir(archiveFile, destDir, archiveType); err != nil {
					span.RecordError(err)
					span.End()
					e.archivesFailed.Add(ctx, 1,
						metric.WithAttributes(
							attribute.String("error_type", "extract_failed"),
							attribute.String("archive_type", string(archiveType)),
						),
					)
					return T.Unit{}, err
				}

				if e.DeleteAfter {
					if err := os.Remove(archiveFile); err != nil {
						e.Logger.Warnw(
							"Failed to delete archive file",
							"archive",
							archiveFile,
							"error",
							err,
						)
					} else {
						e.Logger.Infow("Deleted archive file", "archive", archiveFile)
					}
				}

				span.End()
			}
		}
		return T.Unit{}, nil
	})
}

func (e *Extractor) findAllArchiveFilesRecursive(dir string) ([]string, error) {
	var archiveFiles []string

	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if !d.IsDir() && getArchiveType(d.Name()) != UnknownType {
			archiveFiles = append(archiveFiles, path)
		}

		return nil
	})

	return archiveFiles, err
}

func (e *Extractor) extractToDir(archivePath, destDir string, archiveType ArchiveType) error {
	switch archiveType {
	case ZipType:
		return e.extractZip(archivePath, destDir)
	case TarType:
		return e.extractTar(archivePath, destDir)
	case TarGzType, TgzType:
		return e.extractTarGz(archivePath, destDir)
	default:
		return fmt.Errorf("unsupported archive type: %s", archiveType)
	}
}

func (e *Extractor) extractZip(zipPath, destDir string) error {
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
		cleanDestPath := filepath.Clean(destPath)
		if !strings.HasPrefix(cleanDestPath, filepath.Clean(destDir)+string(filepath.Separator)) {
			return fmt.Errorf("illegal file path: %s", destPath)
		}

		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(cleanDestPath, f.Mode()|0o700); err != nil { // Preserve mode, ensure dir executable
				return fmt.Errorf("failed to create directory %s: %w", cleanDestPath, err)
			}
			continue
		}

		if err := os.MkdirAll(filepath.Dir(cleanDestPath), os.ModePerm); err != nil {
			return fmt.Errorf("failed to create parent directory for %s: %w", cleanDestPath, err)
		}

		rc, err := f.Open()
		if err != nil {
			return fmt.Errorf("failed to open file %s in zip: %w", f.Name, err)
		}

		destFile, err := os.OpenFile(cleanDestPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
		if err != nil {
			rc.Close()
			return fmt.Errorf("failed to create file %s: %w", cleanDestPath, err)
		}

		n, err := io.Copy(destFile, rc)

		destFile.Close()
		rc.Close()

		if err != nil {
			return fmt.Errorf("failed to copy file %s: %w", f.Name, err)
		}

		// Preserve timestamp
		if err := os.Chtimes(cleanDestPath, f.Modified, f.Modified); err != nil {
			e.Logger.Warnw("Failed to set timestamp", "file", cleanDestPath, "error", err)
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

		e.Logger.Debugw("File extracted", "file", f.Name, "dest", cleanDestPath)
	}

	e.Logger.Infow("Zip extraction completed", "zip", zipPath, "files_extracted", len(r.File))
	return nil
}

func (e *Extractor) extractTar(tarPath, destDir string) error {
	file, err := os.Open(tarPath)
	if err != nil {
		return fmt.Errorf("failed to open tar %s: %w", tarPath, err)
	}
	defer file.Close()

	tr := tar.NewReader(file)
	return e.extractTarReader(tr, destDir, tarPath)
}

func (e *Extractor) extractTarGz(tarGzPath, destDir string) error {
	file, err := os.Open(tarGzPath)
	if err != nil {
		return fmt.Errorf("failed to open tar.gz %s: %w", tarGzPath, err)
	}
	defer file.Close()

	gzr, err := gzip.NewReader(file)
	if err != nil {
		return fmt.Errorf("failed to create gzip reader for %s: %w", tarGzPath, err)
	}
	defer gzr.Close()

	tr := tar.NewReader(gzr)
	return e.extractTarReader(tr, destDir, tarGzPath)
}

func (e *Extractor) extractTarReader(tr *tar.Reader, destDir, archivePath string) error {
	startTime := time.Now()
	e.currentArchive = archivePath

	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("failed to read tar header: %w", err)
		}

		e.currentFile = header.Name
		e.updateDescription()

		destPath := filepath.Join(destDir, header.Name)
		cleanDestPath := filepath.Clean(destPath)
		if !strings.HasPrefix(cleanDestPath, filepath.Clean(destDir)+string(filepath.Separator)) {
			return fmt.Errorf("illegal file path: %s", destPath)
		}

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(cleanDestPath, os.FileMode(header.Mode)|0o700); err != nil {
				return fmt.Errorf("failed to create directory %s: %w", cleanDestPath, err)
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(cleanDestPath), os.ModePerm); err != nil {
				return fmt.Errorf(
					"failed to create parent directory for %s: %w",
					cleanDestPath,
					err,
				)
			}
			destFile, err := os.OpenFile(
				cleanDestPath,
				os.O_WRONLY|os.O_CREATE|os.O_TRUNC,
				os.FileMode(header.Mode),
			)
			if err != nil {
				return fmt.Errorf("failed to create file %s: %w", cleanDestPath, err)
			}
			n, err := io.Copy(destFile, tr)
			destFile.Close()
			if err != nil {
				return fmt.Errorf("failed to copy file %s: %w", header.Name, err)
			}
			e.bytesTotal.Add(context.Background(), n)
			e.filesTotal.Add(context.Background(), 1)
			e.ExtractedFiles.Add(1)
		case tar.TypeSymlink:
			// Sanitize link target
			targetPath := filepath.Clean(filepath.Join(destDir, header.Linkname))
			if !strings.HasPrefix(targetPath, filepath.Clean(destDir)+string(filepath.Separator)) {
				e.Logger.Warnw(
					"Skipping illegal symlink target",
					"symlink",
					cleanDestPath,
					"target",
					header.Linkname,
				)
				continue
			}
			if err := os.Symlink(header.Linkname, cleanDestPath); err != nil {
				return fmt.Errorf("failed to create symlink %s: %w", cleanDestPath, err)
			}
		default:
			e.Logger.Debugw(
				"Skipping unsupported type",
				"type",
				header.Typeflag,
				"name",
				header.Name,
			)
			continue
		}

		// Preserve timestamps
		if err := os.Chtimes(cleanDestPath, header.AccessTime, header.ModTime); err != nil {
			e.Logger.Warnw("Failed to set timestamp", "file", cleanDestPath, "error", err)
		}

		e.updateDescription()
	}

	durationMs := time.Since(startTime).Milliseconds()
	e.fileDuration.Record(context.Background(), durationMs,
		metric.WithAttributes(
			attribute.String("status", "success"),
			attribute.Bool("nested", false),
		),
	)

	e.Logger.Infow("Tar extraction completed", "archive", archivePath)
	return nil
}
