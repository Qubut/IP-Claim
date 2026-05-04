package pipeline

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/destel/rill"
)

// streamArchive emits an XMLEntry for every XML payload found in r,
// recursing into nested archives. The caller owns r; internal readers
// (gzip, tar, zip, spool file) are closed here.
func streamArchive(
	ctx context.Context,
	archiveName string,
	r io.Reader,
	kind ArchiveKind,
	cfg WalkConfig,
) rill.Stream[XMLEntry] {
	return rill.Generate(func(send func(XMLEntry), sendErr func(error)) {
		if err := walkArchive(ctx, archiveName, r, kind, cfg, send, sendErr); err != nil {
			sendErr(err)
		}
	})
}

// walkArchive dispatches on kind. Returning an error aborts the whole
// archive; sendErr aborts only the offending entry.
func walkArchive(
	ctx context.Context,
	archiveName string,
	r io.Reader,
	kind ArchiveKind,
	cfg WalkConfig,
	send func(XMLEntry),
	sendErr func(error),
) error {
	switch kind {
	case KindTarGz:
		gzr, err := gzip.NewReader(r)
		if err != nil {
			return fmt.Errorf("gzip open %s: %w", archiveName, err)
		}
		defer func() { _ = gzr.Close() }()
		return walkTar(ctx, archiveName, tar.NewReader(gzr), cfg, send, sendErr)
	case KindTar:
		return walkTar(ctx, archiveName, tar.NewReader(r), cfg, send, sendErr)
	case KindZip:
		return walkZipFromReader(ctx, archiveName, r, cfg, send, sendErr)
	default:
		return fmt.Errorf("unsupported archive kind for %s", archiveName)
	}
}

// walkTar streams entries from tr. The current entry must be fully
// consumed before the next tr.Next() call.
func walkTar(
	ctx context.Context,
	archiveName string,
	tr *tar.Reader,
	cfg WalkConfig,
	send func(XMLEntry),
	sendErr func(error),
) error {
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("tar header in %s: %w", archiveName, err)
		}
		if hdr.Typeflag != tar.TypeReg {
			continue
		}
		if err := dispatchEntry(ctx, archiveName, hdr.Name, hdr.Size, tr, cfg, send, sendErr); err != nil {
			return err
		}
	}
}

// walkZipFromReader spools r to a temp file (zip needs random access)
// and walks each entry. Tar paths never touch disk.
func walkZipFromReader(
	ctx context.Context,
	archiveName string,
	r io.Reader,
	cfg WalkConfig,
	send func(XMLEntry),
	sendErr func(error),
) error {
	tmp, err := os.CreateTemp(resolvedSpoolDir(cfg.SpoolDir), "epo-spool-*.zip")
	if err != nil {
		return fmt.Errorf("spool create for %s: %w", archiveName, err)
	}
	defer func() { _ = os.Remove(tmp.Name()) }()
	defer func() { _ = tmp.Close() }()

	size, err := io.Copy(tmp, r)
	if err != nil {
		return fmt.Errorf("spool copy for %s: %w", archiveName, err)
	}
	if _, err := tmp.Seek(0, io.SeekStart); err != nil {
		return err
	}
	zr, err := zip.NewReader(tmp, size)
	if err != nil {
		return fmt.Errorf("zip open %s: %w", archiveName, err)
	}
	for _, f := range zr.File {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if f.FileInfo().IsDir() {
			continue
		}
		if err := streamZipEntry(ctx, archiveName, f, cfg, send, sendErr); err != nil {
			return err
		}
	}
	return nil
}

func streamZipEntry(
	ctx context.Context,
	archiveName string,
	f *zip.File,
	cfg WalkConfig,
	send func(XMLEntry),
	sendErr func(error),
) error {
	rc, err := f.Open()
	if err != nil {
		sendErr(fmt.Errorf("zip entry open %s/%s: %w", archiveName, f.Name, err))
		return nil
	}
	defer func() { _ = rc.Close() }()
	return dispatchEntry(ctx, archiveName, f.Name, int64(f.UncompressedSize64), rc, cfg, send, sendErr) //nolint:gosec // zip entry size is bounded by zip spec
}

// dispatchEntry emits XML entries, recurses into nested archives, and
// drains anything else.
func dispatchEntry(
	ctx context.Context,
	archiveName, entryName string,
	size int64,
	r io.Reader,
	cfg WalkConfig,
	send func(XMLEntry),
	sendErr func(error),
) error {
	switch {
	case cfg.selector()(entryName):
		reader, closeKept, err := cfg.teeIfKeep(archiveName, entryName, r)
		if err != nil {
			sendErr(err)
			return nil
		}
		// Block until the consumer closes the entry to preserve backpressure.
		done := make(chan struct{})
		send(XMLEntry{
			ArchiveName: archiveName,
			Name:        entryName,
			Size:        size,
			Reader:      reader,
			closer: func() error {
				close(done)
				return closeKept()
			},
		})
		select {
		case <-done:
		case <-ctx.Done():
			return ctx.Err()
		}
		return nil

	case DetectKind(entryName) != KindUnknown:
		if err := walkArchive(ctx, archiveName+"!"+entryName, r,
			DetectKind(entryName), cfg, send, sendErr); err != nil {
			sendErr(fmt.Errorf("nested %s: %w", entryName, err))
		}
		return nil

	default:
		// Drain so tar.Reader can advance.
		_, _ = io.Copy(io.Discard, r)
		return nil
	}
}

// resolvedSpoolDir returns d (created if missing) or os.TempDir() if empty.
func resolvedSpoolDir(d string) string {
	if d == "" {
		return os.TempDir()
	}
	_ = os.MkdirAll(d, 0o750)
	return filepath.Clean(d)
}
