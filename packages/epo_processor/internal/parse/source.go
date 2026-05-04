package parse

import (
	"compress/gzip"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// RecordSource yields PatentRecord values one at a time. Implementations
// must be safe for sequential use; concurrent access is not required.
//
// Next reports (true, nil) for a populated record, (false, nil) on EOF,
// and (false, err) on any other failure. Close releases the underlying
// reader.
type RecordSource interface {
	Next(*PatentRecord) (bool, error)
	Close() error
}

// OpenRecordSource opens path as a RecordSource. The format is selected
// by extension: .parquet → Parquet; .csv or .csv.gz → CSV.
func OpenRecordSource(path string) (RecordSource, error) {
	switch ext := strings.ToLower(filepath.Ext(path)); {
	case ext == ".parquet":
		return openParquetSource(path)
	case ext == ".csv":
		return openCSVSource(path, false)
	case ext == ".gz" && strings.HasSuffix(strings.ToLower(path), ".csv.gz"):
		return openCSVSource(path, true)
	default:
		return nil, fmt.Errorf("unsupported record source extension for %s", path)
	}
}

// --- CSV ---------------------------------------------------------------

// csvSource decodes the columns produced by the legacy CSV writer:
// patent_id, status, cpc_list, citations, family_patents.
// List cells are ";"-joined; citation cells are "<id> (<cats>)" per entry.
type csvSource struct {
	f      *os.File
	gz     io.Closer
	reader *csv.Reader
}

func openCSVSource(path string, gzipped bool) (*csvSource, error) {
	f, err := os.Open(path) //nolint:gosec // path is user config
	if err != nil {
		return nil, err
	}
	var r io.Reader = f
	var gz io.Closer
	if gzipped {
		zr, err := gzip.NewReader(f)
		if err != nil {
			_ = f.Close()
			return nil, err
		}
		r, gz = zr, zr
	}
	cr := csv.NewReader(r)
	cr.FieldsPerRecord = -1 // tolerate header drift
	if _, err := cr.Read(); err != nil {
		_ = f.Close()
		return nil, fmt.Errorf("csv header: %w", err)
	}
	return &csvSource{f: f, gz: gz, reader: cr}, nil
}

func (s *csvSource) Next(rec *PatentRecord) (bool, error) {
	row, err := s.reader.Read()
	if err == io.EOF {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if len(row) < 5 {
		return false, fmt.Errorf("csv row has %d columns, want ≥5", len(row))
	}
	rec.PatentID = row[0]
	rec.Status = row[1]
	rec.CPCList = splitList(row[2])
	rec.Citations = parseCitationsCSV(row[3])
	rec.FamilyPatents = splitList(row[4])
	return true, nil
}

func (s *csvSource) Close() error {
	if s.gz != nil {
		_ = s.gz.Close()
	}
	return s.f.Close()
}

// splitList splits a ";"-joined cell, dropping empty fragments.
func splitList(cell string) []string {
	if cell == "" {
		return nil
	}
	parts := strings.Split(cell, ";")
	out := parts[:0]
	for _, p := range parts {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// parseCitationsCSV decodes a citations cell of the form
// "US3270412A (X);FR1459892A ()". Each entry yields one Citation; the
// "(...)" suffix is split on whitespace into Categories.
func parseCitationsCSV(cell string) []Citation {
	if cell == "" {
		return nil
	}
	entries := strings.Split(cell, ";")
	out := make([]Citation, 0, len(entries))
	for _, e := range entries {
		e = strings.TrimSpace(e)
		if e == "" {
			continue
		}
		id, cats := splitCitationEntry(e)
		out = append(out, Citation{CitedID: id, Categories: cats})
	}
	return out
}

func splitCitationEntry(s string) (string, []string) {
	open := strings.LastIndexByte(s, '(')
	closeIdx := strings.LastIndexByte(s, ')')
	if open < 0 || closeIdx < open {
		return strings.TrimSpace(s), nil
	}
	id := strings.TrimSpace(s[:open])
	inner := strings.TrimSpace(s[open+1 : closeIdx])
	if inner == "" {
		return id, nil
	}
	// Categories may be space-separated ("X Y"), comma-separated ("X,Y"),
	// or a mix ("X,Y Z"). Normalize both delimiters to spaces first.
	normalized := strings.ReplaceAll(inner, ",", " ")
	raw := strings.Fields(normalized)
	cats := make([]string, 0, len(raw))
	for _, c := range raw {
		c = strings.TrimSpace(c)
		if c != "" {
			cats = append(cats, c)
		}
	}
	return id, cats
}
