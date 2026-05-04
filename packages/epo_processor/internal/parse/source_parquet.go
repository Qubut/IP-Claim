package parse

import (
	"io"
	"os"

	"github.com/parquet-go/parquet-go"
)

// parquetSource reads PatentRecord rows from a Parquet file.
//
// Memory is bounded by batchSize × row width; the underlying
// GenericReader pulls one row group at a time.
type parquetSource struct {
	f       *os.File
	reader  *parquet.GenericReader[PatentRecord]
	batch   []PatentRecord
	pos     int
	loaded  int
	drained bool
}

const parquetBatchSize = 1024

func openParquetSource(path string) (*parquetSource, error) {
	f, err := os.Open(path) //nolint:gosec // path is user config
	if err != nil {
		return nil, err
	}
	return &parquetSource{
		f:      f,
		reader: parquet.NewGenericReader[PatentRecord](f),
		batch:  make([]PatentRecord, parquetBatchSize),
	}, nil
}

func (s *parquetSource) Next(rec *PatentRecord) (bool, error) {
	if s.pos >= s.loaded {
		if s.drained {
			return false, nil
		}
		n, err := s.reader.Read(s.batch)
		if err != nil && err != io.EOF {
			return false, err
		}
		if err == io.EOF {
			s.drained = true
		}
		if n == 0 {
			return false, nil
		}
		s.loaded = n
		s.pos = 0
	}
	*rec = s.batch[s.pos]
	s.pos++
	return true, nil
}

func (s *parquetSource) Close() error {
	_ = s.reader.Close()
	return s.f.Close()
}
