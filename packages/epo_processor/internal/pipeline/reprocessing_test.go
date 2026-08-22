package pipeline

import (
	"context"
	"crypto/sha1" //nolint:gosec // EPO BDDS mandates SHA-1; test mirrors that
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

// recordingJanitor records the names of archives marked complete. It is
// goroutine-safe because OnArchiveDone runs from concurrent opener workers.
type recordingJanitor struct {
	mu   sync.Mutex
	done []string
}

func (r *recordingJanitor) OnArchiveDone(_ context.Context, j ArchiveJob, _ string, _ bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.done = append(r.done, j.Name)
}

func (r *recordingJanitor) OnEntryDone(context.Context, XMLEntry, string) {}

func (r *recordingJanitor) names() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.done...)
}

func contains(haystack []string, want string) bool {
	for _, s := range haystack {
		if s == want {
			return true
		}
	}
	return false
}

// A checksum mismatch must NOT abort the run: records are kept and the
// archive is marked complete (so it is never re-downloaded).
func TestPipeline_ChecksumMismatch_KeepsRecordsAndMarksComplete(t *testing.T) {
	archive := makeTarGz(t, map[string]string{"docs/a.xml": xmlOne})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/gzip")
		_, _ = w.Write(archive)
	}))
	defer srv.Close()

	src := &StaticListSource{Jobs: []ArchiveJob{
		{Name: "a.tar.gz", URL: srv.URL, Checksum: "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"},
	}}
	jan := &recordingJanitor{}
	opener := NewHTTPOpener(srv.Client(), 0, true, WalkConfig{SpoolDir: t.TempDir()})
	opener.Janitor = jan
	sink := &MemorySink{}

	p, err := New(
		WithSource(src),
		WithOpener(opener),
		WithExtractor(NewXMLStreamExtractor()),
		WithSink(sink),
		WithBatchSize(10),
		WithBatchTimeout(50*time.Millisecond),
	)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := p.Run(ctx); err != nil {
		t.Fatalf("run should not fail on checksum mismatch, got: %v", err)
	}
	if len(sink.Records) != 2 {
		t.Fatalf("want 2 kept records despite mismatch, got %d", len(sink.Records))
	}
	if !contains(jan.names(), "a.tar.gz") {
		t.Fatalf("archive should be marked complete after mismatch, got %v", jan.names())
	}
}

// A matching checksum still completes normally (guards against the warn
// path firing on valid archives).
func TestPipeline_ChecksumMatch_Completes(t *testing.T) {
	archive := makeTarGz(t, map[string]string{"docs/a.xml": xmlOne})
	sum := sha1.Sum(archive) //nolint:gosec // see import note

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(archive)
	}))
	defer srv.Close()

	src := &StaticListSource{Jobs: []ArchiveJob{
		{Name: "a.tar.gz", URL: srv.URL, Checksum: hex.EncodeToString(sum[:])},
	}}
	jan := &recordingJanitor{}
	opener := NewHTTPOpener(srv.Client(), 0, true, WalkConfig{SpoolDir: t.TempDir()})
	opener.Janitor = jan
	sink := &MemorySink{}

	p, _ := New(
		WithSource(src),
		WithOpener(opener),
		WithExtractor(NewXMLStreamExtractor()),
		WithSink(sink),
		WithBatchSize(10),
		WithBatchTimeout(50*time.Millisecond),
	)
	if err := p.Run(context.Background()); err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(sink.Records) != 2 {
		t.Fatalf("want 2 records, got %d", len(sink.Records))
	}
	if !contains(jan.names(), "a.tar.gz") {
		t.Fatalf("archive should be marked complete, got %v", jan.names())
	}
}

// A download that always fails (HTTP 500) must NOT abort the run: a sibling
// good archive still produces records and is marked complete, while the bad
// archive is left unmarked so it retries next run.
func TestPipeline_FailingDownload_DoesNotAbort_SiblingSucceeds(t *testing.T) {
	archive := makeTarGz(t, map[string]string{"docs/a.xml": xmlOne})

	mux := http.NewServeMux()
	mux.HandleFunc("/bad", func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	})
	mux.HandleFunc("/good", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/gzip")
		_, _ = w.Write(archive)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	src := &StaticListSource{Jobs: []ArchiveJob{
		{Name: "bad.tar.gz", URL: srv.URL + "/bad"},
		{Name: "good.tar.gz", URL: srv.URL + "/good"},
	}}
	jan := &recordingJanitor{}
	opener := NewHTTPOpener(srv.Client(), 0, false, WalkConfig{SpoolDir: t.TempDir()})
	opener.Janitor = jan
	sink := &MemorySink{}

	p, err := New(
		WithSource(src),
		WithOpener(opener),
		WithExtractor(NewXMLStreamExtractor()),
		WithSink(sink),
		WithBatchSize(10),
		WithBatchTimeout(50*time.Millisecond),
	)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := p.Run(ctx); err != nil {
		t.Fatalf("run should not fail when one archive download fails, got: %v", err)
	}
	if len(sink.Records) != 2 {
		t.Fatalf("want 2 records from the good archive, got %d", len(sink.Records))
	}
	names := jan.names()
	if !contains(names, "good.tar.gz") {
		t.Fatalf("good archive should be marked complete, got %v", names)
	}
	if contains(names, "bad.tar.gz") {
		t.Fatalf("failed archive must stay unmarked so it retries next run, got %v", names)
	}
}
