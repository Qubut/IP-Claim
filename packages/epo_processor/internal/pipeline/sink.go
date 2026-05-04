package pipeline

import (
	"context"
	"fmt"
	"os"
	"sync"

	"github.com/parquet-go/parquet-go"
)

// DefaultRowGroupSize is the default record count threshold at which
// [ParquetSink] flushes the current row group to disk.
const DefaultRowGroupSize = 50_000

// ParquetSink writes [PatentRecord] batches to a single Parquet file.
//
// Write must be called from a single goroutine (the pipeline already
// guarantees this). The sink calls Flush() once the count of pending
// records crosses flushEvery to bound memory; without it the writer
// holds every record until Close.
//
// The file footer is only written at Close — the file remains
// unreadable as Parquet until then. For crash-recoverable output use
// a sharded sink (one file per archive).
type ParquetSink struct {
	path       string
	file       *os.File
	writer     *parquet.GenericWriter[PatentRecord]
	once       sync.Once
	flushEvery int
	pending    int
}

// NewParquetSink creates path and returns a sink writing the
// [PatentRecord] schema. flushEvery <= 0 selects [DefaultRowGroupSize].
func NewParquetSink(path string, flushEvery int) (*ParquetSink, error) {
	if flushEvery <= 0 {
		flushEvery = DefaultRowGroupSize
	}
	f, err := os.Create(path) //nolint:gosec // path from config
	if err != nil {
		return nil, fmt.Errorf("create parquet %s: %w", path, err)
	}
	return &ParquetSink{
		path:       path,
		file:       f,
		writer:     parquet.NewGenericWriter[PatentRecord](f),
		flushEvery: flushEvery,
	}, nil
}

// Write appends batch. When pending records cross flushEvery, the
// current row group is finalised to disk.
func (s *ParquetSink) Write(_ context.Context, batch []PatentRecord) error {
	if _, err := s.writer.Write(batch); err != nil {
		return err
	}
	s.pending += len(batch)
	if s.pending >= s.flushEvery {
		if err := s.writer.Flush(); err != nil {
			return fmt.Errorf("parquet flush: %w", err)
		}
		s.pending = 0
	}
	return nil
}

// Close flushes the Parquet footer and closes the underlying file.
// Must be called or the file will be unreadable. Idempotent.
func (s *ParquetSink) Close() error {
	var err error
	s.once.Do(func() {
		if cerr := s.writer.Close(); cerr != nil {
			err = cerr
		}
		if cerr := s.file.Close(); cerr != nil && err == nil {
			err = cerr
		}
	})
	return err
}

// MemorySink is an in-process [RecordSink] for tests. Goroutine-safe.
type MemorySink struct {
	mu      sync.Mutex
	Records []PatentRecord
}

func (m *MemorySink) Write(_ context.Context, batch []PatentRecord) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.Records = append(m.Records, batch...)
	return nil
}

// Close is a no-op for MemorySink.
func (m *MemorySink) Close() error { return nil }
