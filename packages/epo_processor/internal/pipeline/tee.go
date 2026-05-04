package pipeline

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	A "github.com/IBM/fp-go/v2/array"
	F "github.com/IBM/fp-go/v2/function"
)

// noopCloser is reused by every "keep disabled" path.
var noopCloser = func() error { return nil }

// optionalTee returns r unchanged when enabled is false; otherwise it
// mirrors r to disk.
func optionalTee(enabled bool, archive, entry, baseDir string, r io.Reader) (io.Reader, func() error, error) {
	if !enabled {
		return r, noopCloser, nil
	}
	return tee(archive, entry, baseDir, r)
}

// tee creates baseDir/archiveSubdir(archive)/entry and tees r into it.
// The returned closer must be invoked exactly once.
func tee(archive, entry, baseDir string, r io.Reader) (io.Reader, func() error, error) {
	if baseDir == "" {
		return nil, nil, fmt.Errorf("keep enabled but destination dir is empty")
	}
	subdir := filepath.Join(baseDir, archiveSubdir(archive))
	dst := filepath.Join(subdir, entry)
	cleanRoot := filepath.Clean(subdir) + string(filepath.Separator)
	if !strings.HasPrefix(filepath.Clean(dst), cleanRoot) {
		return nil, nil, fmt.Errorf("illegal extracted path: %s", dst)
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o750); err != nil {
		return nil, nil, err
	}
	f, err := os.Create(dst) //nolint:gosec // dst is sanitized above
	if err != nil {
		return nil, nil, err
	}
	return io.TeeReader(r, f), f.Close, nil
}

// archiveExtensions are stripped from each chain segment.
var archiveExtensions = []string{".tar.gz", ".tgz", ".tar", ".zip"}

// pathRepl neutralises filesystem-hostile characters in a chain segment.
var pathRepl = strings.NewReplacer("/", "_", "\\", "_", ":", "_")

// archiveSubdir maps a bang-chained archive name to a nested directory.
//
// "data/all.tar!2011.tar.gz" → "all/2011".
func archiveSubdir(archive string) string {
	parts := F.Pipe2(
		strings.Split(archive, "!"),
		A.Map(cleanSegment),
		A.Filter(nonEmpty),
	)
	return filepath.Join(parts...)
}

func cleanSegment(seg string) string {
	base := filepath.Base(seg)
	lower := strings.ToLower(base)
	for _, ext := range archiveExtensions {
		if strings.HasSuffix(lower, ext) {
			return pathRepl.Replace(base[:len(base)-len(ext)])
		}
	}
	return pathRepl.Replace(base)
}

func nonEmpty(s string) bool { return s != "" && s != "." }
