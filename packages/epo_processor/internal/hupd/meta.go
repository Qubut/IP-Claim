package hupd

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/apache/arrow/go/v18/arrow"
	"github.com/apache/arrow/go/v18/arrow/array"
	"github.com/apache/arrow/go/v18/arrow/ipc"
	"github.com/apache/arrow/go/v18/arrow/memory"
)

// EnsureMeta downloads the HUPD metadata Feather file from metaURL to
// metaPath unless the file already exists on disk. The download is written
// to a sibling temp file and atomically renamed on success, so a partial
// download never leaves a corrupt file at metaPath.
//
// bar and mu are used for progress reporting: mu must be held while calling
// bar methods so the caller's progress bar is updated from a single
// goroutine. log may be nil (no progress lines are emitted in that case).
func EnsureMeta(
	ctx context.Context,
	metaPath, metaURL string,
	bar ProgressReporter,
	mu *sync.Mutex,
	log *slog.Logger,
) error {
	if _, err := os.Stat(metaPath); err == nil {
		return nil // already on disk
	}
	if err := os.MkdirAll(filepath.Dir(metaPath), 0o750); err != nil {
		return fmt.Errorf("mkdir for feather: %w", err)
	}

	log.Info("analyze: downloading HUPD metadata feather", "url", metaURL, "dst", metaPath)
	mu.Lock()
	bar.Describe("downloading HUPD meta")
	mu.Unlock()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, metaURL, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("download feather: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download feather: HTTP %d %s", resp.StatusCode, resp.Status)
	}

	tmp, err := os.CreateTemp(filepath.Dir(metaPath), ".hupd-meta-*.feather")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()

	var downloaded int64
	var lastLogged, lastBar time.Time
	start := time.Now()
	buf := make([]byte, 1<<20)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := tmp.Write(buf[:n]); werr != nil {
				_ = tmp.Close()
				_ = os.Remove(tmpPath)
				return werr
			}
			downloaded += int64(n)
			now := time.Now()
			if now.Sub(lastBar) >= BarEvery {
				lastBar = now
				mib := downloaded >> 20
				rate := float64(mib) / now.Sub(start).Seconds()
				mu.Lock()
				bar.Describe(fmt.Sprintf("downloading HUPD meta \u2014 %s MiB \u00b7 %s",
					FmtCount(mib), FmtRate(rate)))
				mu.Unlock()
			}
			if now.Sub(lastLogged) >= ProgressEvery {
				lastLogged = now
				log.Info("analyze: download progress", "mib", downloaded>>20)
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			_ = tmp.Close()
			_ = os.Remove(tmpPath)
			return fmt.Errorf("download feather body: %w", readErr)
		}
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	if err := os.Rename(tmpPath, metaPath); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	log.Info("analyze: feather downloaded", "path", metaPath, "mib", downloaded>>20)
	return nil
}

// ScanMeta builds the normalised-ID → []path index from the HUPD
// Feather metadata file. If metaPath does not exist it is downloaded
// first via [EnsureMeta].
//
// Columns consumed: application_number, patent_number, publication_number,
// filing_date. Per-row paths are inferred as hupdDir/<year>/<year>/<app>.json
// where year is derived from the filing_date column.
//
// Each unique normalised ID ([NormalizeUSID]) maps to one or more paths;
// multiple paths arise when the same patent appears under more than one
// application number in the index.
//
// bar and mu are used for progress reporting (see [EnsureMeta]).
func ScanMeta(
	ctx context.Context,
	metaPath, metaURL, hupdDir string,
	bar ProgressReporter,
	mu *sync.Mutex,
	log *slog.Logger,
) (map[string][]string, error) {
	if err := EnsureMeta(ctx, metaPath, metaURL, bar, mu, log); err != nil {
		return nil, err
	}

	f, err := os.Open(metaPath) //nolint:gosec // path is user-supplied config, not tainted input
	if err != nil {
		return nil, fmt.Errorf("open feather: %w", err)
	}
	defer func() { _ = f.Close() }()

	reader, err := ipc.NewFileReader(f, ipc.WithAllocator(memory.DefaultAllocator))
	if err != nil {
		return nil, fmt.Errorf("read feather header: %w", err)
	}
	defer func() { _ = reader.Close() }()

	schema := reader.Schema()
	must := func(name string) (int, error) {
		if idx := schema.FieldIndices(name); len(idx) > 0 {
			return idx[0], nil
		}
		return 0, fmt.Errorf("feather: missing column %q", name)
	}

	appIdx, err := must("application_number")
	if err != nil {
		return nil, err
	}
	patIdx, err := must("patent_number")
	if err != nil {
		return nil, err
	}
	pubIdx, _ := schema.FieldIndices("publication_number"), false
	hasPub := len(pubIdx) > 0
	var pubIdxVal int
	if hasPub {
		pubIdxVal = pubIdx[0]
	}
	dateIdx, err := must("filing_date")
	if err != nil {
		return nil, err
	}

	log.Info("analyze: feather schema",
		"application_number_type", schema.Field(appIdx).Type,
		"patent_number_type", schema.Field(patIdx).Type,
		"filing_date_type", schema.Field(dateIdx).Type,
		"has_publication_number", hasPub,
		"num_record_batches", reader.NumRecords(),
	)

	mu.Lock()
	bar.Describe("HUPD meta")
	mu.Unlock()

	out := make(map[string][]string, 5_000_000)
	var total int64
	var lastLogged, lastBar time.Time
	start := time.Now()
	epoch := time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC)

	addIDs := func(path string, ids []string) {
		for _, id := range ids {
			if id != "" {
				out[id] = append(out[id], path)
			}
		}
	}

	for i := 0; i < reader.NumRecords(); i++ {
		rec, err := reader.Record(i)
		if err != nil {
			return nil, fmt.Errorf("feather record batch %d: %w", i, err)
		}

		appFn := arrowStringFn(rec.Column(appIdx))
		patFn := arrowStringFn(rec.Column(patIdx))
		dateFn := arrowDateYearFn(rec.Column(dateIdx), epoch)
		pubFn := func(int) string { return "" }
		if hasPub {
			pubFn = arrowStringFn(rec.Column(pubIdxVal))
		}

		for row := 0; row < int(rec.NumRows()); row++ {
			appNum, year := appFn(row), dateFn(row)
			if appNum == "" || year == "" {
				continue
			}
			path := filepath.Join(hupdDir, year, year, appNum+".json")
			addIDs(path, []string{
				NormalizeHUPDPatentNumber(patFn(row)),
				NormalizeHUPDPublication(pubFn(row)),
			})
			total++
		}
		rec.Release()

		now := time.Now()
		if now.Sub(lastBar) >= BarEvery {
			lastBar = now
			rate := float64(total) / now.Sub(start).Seconds()
			mu.Lock()
			bar.Describe(fmt.Sprintf("HUPD meta \u2014 rows %s \u00b7 %s",
				FmtCount(total), FmtRate(rate)))
			mu.Unlock()
		}
		if now.Sub(lastLogged) >= ProgressEvery {
			lastLogged = now
			log.Info("analyze: meta progress", "rows", total, "ids_kept", len(out))
		}
	}

	return out, nil
}

// arrowStringFn returns a row-index → string accessor for utf8, large-utf8,
// or dictionary-encoded string columns. Returns "" for null or unknown types.
func arrowStringFn(col arrow.Array) func(int) string {
	switch c := col.(type) {
	case *array.String:
		return func(i int) string {
			if c.IsNull(i) {
				return ""
			}
			return c.Value(i)
		}
	case *array.LargeString:
		return func(i int) string {
			if c.IsNull(i) {
				return ""
			}
			return c.Value(i)
		}
	case *array.Dictionary:
		switch dict := c.Dictionary().(type) {
		case *array.String:
			return func(i int) string {
				if c.IsNull(i) {
					return ""
				}
				idx := c.GetValueIndex(i)
				if dict.IsNull(idx) {
					return ""
				}
				return dict.Value(idx)
			}
		case *array.LargeString:
			return func(i int) string {
				if c.IsNull(i) {
					return ""
				}
				idx := c.GetValueIndex(i)
				if dict.IsNull(idx) {
					return ""
				}
				return dict.Value(idx)
			}
		}
	}
	return func(int) string { return "" }
}

// arrowDateYearFn returns a row-index → 4-digit year string accessor.
// Handles date32, date64, timestamp (any unit), and string "YYYY-*" fallback.
func arrowDateYearFn(col arrow.Array, epoch time.Time) func(int) string {
	switch c := col.(type) {
	case *array.Date32:
		return func(i int) string {
			if c.IsNull(i) {
				return ""
			}
			return fmt.Sprintf("%04d", epoch.AddDate(0, 0, int(c.Value(i))).Year())
		}
	case *array.Date64:
		return func(i int) string {
			if c.IsNull(i) {
				return ""
			}
			return fmt.Sprintf("%04d", time.UnixMilli(int64(c.Value(i))).UTC().Year())
		}
	case *array.Timestamp:
		dt := c.DataType().(*arrow.TimestampType)
		toYear := func(ts int64) string {
			var sec int64
			switch dt.Unit {
			case arrow.Second:
				sec = ts
			case arrow.Millisecond:
				sec = ts / 1_000
			case arrow.Microsecond:
				sec = ts / 1_000_000
			default: // Nanosecond
				sec = ts / 1_000_000_000
			}
			return fmt.Sprintf("%04d", time.Unix(sec, 0).UTC().Year())
		}
		return func(i int) string {
			if c.IsNull(i) {
				return ""
			}
			return toYear(int64(c.Value(i)))
		}
	default:
		strFn := arrowStringFn(col)
		return func(i int) string {
			if s := strFn(i); len(s) >= 4 {
				return s[:4]
			}
			return ""
		}
	}
}
