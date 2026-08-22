package pipeline

import (
	"context"
	"io"

	"github.com/destel/rill"

	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/parse"
)

// PatentRecord is re-exported from internal/parse for caller convenience.
type PatentRecord = parse.PatentRecord

// ArchiveJob describes a single archive to process.
type ArchiveJob struct {
	Name         string // human-readable label used in log messages and spans
	URL          string // HTTP download URL; empty for local-file sources
	LocalPath    string // path on disk; empty for HTTP sources
	ExpectedSize int64  // byte count from the catalogue; 0 if unknown
	Checksum     string // SHA-1 hex as listed in the catalogue; empty disables verification
}

// ArchiveKind enumerates the supported container formats.
type ArchiveKind uint8

const (
	// KindUnknown indicates an unrecognised archive format.
	KindUnknown ArchiveKind = iota
	// KindTar is a plain .tar archive.
	KindTar
	// KindTarGz is a gzip-compressed .tar archive.
	KindTarGz
	// KindZip is a .zip archive.
	KindZip
)

// XMLEntry is a single XML payload found inside an archive. The walker
// buffers the (small) payload in memory and emits it without holding the
// container stream open, so entries from one archive can be parsed
// concurrently. Calling [XMLEntry.Close] is still required (it is a no-op for
// buffered entries) to keep the consumer contract uniform.
type XMLEntry struct {
	ArchiveName string    // name of the enclosing archive (may be a "!" chain for nested archives)
	Name        string    // path of this entry within the archive
	Size        int64     // payload byte count
	Reader      io.Reader // in-memory entry body; independent of the container stream
	closer      func() error
}

// Close releases the entry. Safe to call on a nil receiver.
func (e *XMLEntry) Close() error {
	if e == nil || e.closer == nil {
		return nil
	}
	return e.closer()
}

// ArchiveSource produces a stream of jobs to process.
type ArchiveSource interface {
	Stream(ctx context.Context) rill.Stream[ArchiveJob]
}

// ArchiveOpener turns one ArchiveJob into a stream of XMLEntry values,
// recursing into nested archives. It owns all I/O, retry, checksum
// verification and temp-file lifecycle.
type ArchiveOpener interface {
	Stream(ctx context.Context, job ArchiveJob) rill.Stream[XMLEntry]
}

// RecordExtractor turns one XMLEntry into a stream of PatentRecord values.
type RecordExtractor interface {
	Stream(ctx context.Context, entry XMLEntry) rill.Stream[PatentRecord]
}

// RecordSink consumes record batches. Implementations need only be safe
// for a single consumer goroutine.
type RecordSink interface {
	Write(ctx context.Context, batch []PatentRecord) error
	Close() error
}

// Janitor receives lifecycle notifications so retention policy lives
// outside openers and sinks.
type Janitor interface {
	OnArchiveDone(ctx context.Context, job ArchiveJob, localPath string, kept bool)
	OnEntryDone(ctx context.Context, entry XMLEntry, localPath string)
}

// NoopJanitor is the default Janitor; it does nothing.
type NoopJanitor struct{}

// OnArchiveDone is a no-op.
func (NoopJanitor) OnArchiveDone(context.Context, ArchiveJob, string, bool) {}

// OnEntryDone is a no-op.
func (NoopJanitor) OnEntryDone(context.Context, XMLEntry, string) {}
