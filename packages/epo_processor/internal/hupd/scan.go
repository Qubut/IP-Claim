package hupd

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	A "github.com/IBM/fp-go/v2/array"
	IOResF "github.com/IBM/fp-go/v2/idiomatic/ioresult/file"
	P "github.com/IBM/fp-go/v2/predicate"
	"github.com/destel/rill"
)

// Header captures the only HUPD JSON fields needed for ID matching.
type Header struct {
	PatentNumber      string `json:"patent_number"`
	PublicationNumber string `json:"publication_number"`
}

// ScanIDs walks dir recursively, reads the patent_number and
// publication_number header fields from each .json file in parallel
// (up to workers goroutines), and returns a normalised-ID → []path
// index identical in shape to the one produced by [ScanMeta].
//
// Use ScanIDs when the Feather metadata file is unavailable or when
// only a subset of the full HUPD dataset is present on disk.
//
// bar and mu are used for progress reporting (see [EnsureMeta]).
// log may be nil.
func ScanIDs(
	ctx context.Context,
	dir string,
	workers int,
	bar ProgressReporter,
	mu *sync.Mutex,
	log *slog.Logger,
) (map[string][]string, error) {
	type pathRec struct {
		path string
		ids  [2]string
	}

	paths := make(chan rill.Try[string], 1024)
	walkErr := make(chan error, 1)
	go func() {
		defer close(paths)
		walkErr <- filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if ctx.Err() != nil {
				return ctx.Err()
			}
			if d.IsDir() || !strings.EqualFold(filepath.Ext(p), ".json") {
				return nil
			}
			select {
			case paths <- rill.Wrap(p, nil):
			case <-ctx.Done():
				return ctx.Err()
			}
			return nil
		})
	}()

	recs := rill.Map(paths, workers, func(p string) (pathRec, error) {
		ids, err := ReadIDs(p)
		return pathRec{path: p, ids: ids}, err
	})

	out := make(map[string][]string, 1<<16)
	var (
		filesSeen  int64
		lastLogged time.Time
		lastBar    time.Time
	)
	start := time.Now()
	mu.Lock()
	bar.Describe("HUPD scan")
	mu.Unlock()

	if err := rill.ForEach(recs, 1, func(rec pathRec) error {
		filesSeen++
		for _, id := range A.Filter(P.IsNonZero[string]())(rec.ids[:]) {
			out[id] = append(out[id], rec.path)
		}
		now := time.Now()
		if now.Sub(lastBar) >= BarEvery {
			lastBar = now
			rate := float64(filesSeen) / now.Sub(start).Seconds()
			mu.Lock()
			bar.Describe(fmt.Sprintf("HUPD scan \u2014 files %s \u00b7 ids %s \u00b7 %s",
				FmtCount(filesSeen), FmtCount(int64(len(out))), FmtRate(rate)))
			mu.Unlock()
		}
		if now.Sub(lastLogged) >= ProgressEvery {
			lastLogged = now
			log.Info("analyze: HUPD progress", "files_seen", filesSeen, "ids_kept", len(out))
		}
		return nil
	}); err != nil {
		return nil, err
	}
	if err := <-walkErr; err != nil {
		return nil, err
	}
	return out, nil
}

// ReadIDs reads one HUPD JSON and returns its two normalized US identifiers
// (patent_number, publication_number). Either may be empty.
func ReadIDs(path string) ([2]string, error) {
	return IOResF.Write[[2]string, *os.File](IOResF.Open(path))(func(f *os.File) func() ([2]string, error) {
		return func() ([2]string, error) {
			var h Header
			if err := json.NewDecoder(f).Decode(&h); err != nil {
				return [2]string{}, fmt.Errorf("%s: %w", path, err)
			}
			return [2]string{
				NormalizeHUPDPatentNumber(h.PatentNumber),
				NormalizeHUPDPublication(h.PublicationNumber),
			}, nil
		}
	})()
}
