package pipeline

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestKeepArchive verifies the HTTP body is teed to disk under ArchiveDir
// while the walker consumes it — and that both happen successfully.
func TestKeepArchive_WritesArchiveFile(t *testing.T) {
	archive := makeTarGz(t, map[string]string{"a.xml": xmlOne})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(archive)
	}))
	defer srv.Close()

	archiveDir := t.TempDir()
	opener := NewHTTPOpener(srv.Client(), 0, false, WalkConfig{SpoolDir: t.TempDir()})
	opener.KeepArchive = true
	opener.ArchiveDir = archiveDir

	src := &StaticListSource{Jobs: []ArchiveJob{{Name: "kept.tar.gz", URL: srv.URL}}}
	sink := &MemorySink{}
	p, _ := New(
		WithSource(src),
		WithOpener(opener),
		WithExtractor(NewXMLStreamExtractor()),
		WithSink(sink),
		WithBatchSize(10),
		WithBatchTimeout(100*time.Millisecond),
	)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := p.Run(ctx); err != nil {
		t.Fatalf("run: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(archiveDir, "kept.tar.gz")) //nolint:gosec // test helper path
	if err != nil {
		t.Fatalf("read kept archive: %v", err)
	}
	if len(got) != len(archive) {
		t.Fatalf("kept archive size %d != original %d", len(got), len(archive))
	}
	if len(sink.Records) != 2 {
		t.Fatalf("want 2 records (parser still ran), got %d", len(sink.Records))
	}
}

// TestKeepExtracted verifies XML entries are mirrored to disk under
// ExtractedDir while the extractor consumes them.
func TestKeepExtracted_WritesEntryFiles(t *testing.T) {
	archive := makeTarGz(t, map[string]string{"docs/a.xml": xmlOne})
	extDir := t.TempDir()
	src := &StaticListSource{Jobs: []ArchiveJob{{
		Name:      "src.tar.gz",
		LocalPath: writeTemp(t, "src.tar.gz", archive),
	}}}
	opener := NewLocalFileOpener(WalkConfig{
		SpoolDir:      t.TempDir(),
		KeepExtracted: true,
		ExtractedDir:  extDir,
	})
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
	want := filepath.Join(extDir, "src", "docs", "a.xml")
	st, err := os.Stat(want)
	if err != nil {
		t.Fatalf("expected extracted file %s: %v", want, err)
	}
	if st.Size() == 0 {
		t.Fatal("extracted file is empty")
	}
}

// TestProcessHupd_Composition exercises the HUPD-shaped composition:
// match-all selector + NoopExtractor + NoopSink + KeepArchive +
// KeepExtracted. The HUPD payload is a flat .tar of JSON files.
func TestProcessHupd_Composition_StreamsBothToDisk(t *testing.T) {
	tarBytes := makeTar(t, map[string]string{"2010/0001.json": `{"id":"0001"}`})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(tarBytes)
	}))
	defer srv.Close()

	archiveDir := t.TempDir()
	extDir := t.TempDir()
	walk := WalkConfig{
		SpoolDir:      t.TempDir(),
		KeepExtracted: true,
		ExtractedDir:  extDir,
		EntrySelector: func(string) bool { return true },
	}
	opener := NewHTTPOpener(srv.Client(), 0, false, walk)
	opener.KeepArchive = true
	opener.ArchiveDir = archiveDir

	p, _ := New(
		WithSource(&StaticListSource{Jobs: []ArchiveJob{{Name: "hupd.tar", URL: srv.URL}}}),
		WithOpener(opener),
		WithExtractor(NoopExtractor{}),
		WithSink(NoopSink{}),
		WithBatchSize(1),
		WithBatchTimeout(100*time.Millisecond),
	)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := p.Run(ctx); err != nil {
		t.Fatalf("run: %v", err)
	}
	if _, err := os.Stat(filepath.Join(archiveDir, "hupd.tar")); err != nil {
		t.Fatalf("kept archive missing: %v", err)
	}
	want := filepath.Join(extDir, "hupd", "2010", "0001.json")
	if _, err := os.Stat(want); err != nil {
		t.Fatalf("extracted entry missing: %v (looked at %s)", err, want)
	}
}

// TestSanitize_PathTraversalGuard ensures malicious entry names cannot
// escape the configured ExtractedDir via "../".
func TestTee_PathTraversalGuard(t *testing.T) {
	base := t.TempDir()
	_, _, err := tee("arch", "../evil", base, strings.NewReader("x"))
	if err == nil {
		t.Fatal("expected error for path-traversal entry, got nil")
	}
	if !strings.Contains(err.Error(), "illegal extracted path") {
		t.Fatalf("unexpected error: %v", err)
	}
}

// --- helpers ---------------------------------------------------------------

func writeTemp(t *testing.T, name string, data []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write temp: %v", err)
	}
	return path
}
