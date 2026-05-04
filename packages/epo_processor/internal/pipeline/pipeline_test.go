package pipeline

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// --- helpers ---------------------------------------------------------------

const xmlOne = `<?xml version="1.0"?>
<root>
  <exchange-document country="EP" doc-number="1000001" kind="A1" status="published">
    <bibliographic-data>
      <patent-classifications>
        <patent-classification>
          <classification-scheme scheme="CPCI"/>
          <classification-symbol>G06F 17/30</classification-symbol>
        </patent-classification>
      </patent-classifications>
    </bibliographic-data>
  </exchange-document>
  <exchange-document country="EP" doc-number="1000002" kind="A1" status="published"/>
</root>`

// makeTarGz produces a tar.gz archive in memory containing the given
// (filename → content) entries. Used by both unit and end-to-end tests.
func makeTarGz(t *testing.T, files map[string]string) []byte {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for name, content := range files {
		hdr := &tar.Header{
			Name:     name,
			Mode:     0o644,
			Size:     int64(len(content)),
			Typeflag: tar.TypeReg,
			ModTime:  time.Unix(0, 0),
		}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatalf("tar header: %v", err)
		}
		if _, err := tw.Write([]byte(content)); err != nil {
			t.Fatalf("tar write: %v", err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("tar close: %v", err)
	}
	if err := gz.Close(); err != nil {
		t.Fatalf("gzip close: %v", err)
	}
	return buf.Bytes()
}

// makeNestedTarGz builds an outer .tar.gz that contains an inner .tar.gz
// containing the XML files — exercises the recursive walker.
func makeNestedTarGz(t *testing.T, files map[string]string) []byte {
	inner := makeTarGz(t, files)
	return makeTarGz(t, map[string]string{"inner.tar.gz": string(inner)})
}

// makeTar builds a plain (uncompressed) tar — used for HUPD-shaped tests.
func makeTar(t *testing.T, files map[string]string) []byte {
	t.Helper()
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	for name, content := range files {
		hdr := &tar.Header{
			Name: name, Mode: 0o644, Size: int64(len(content)),
			Typeflag: tar.TypeReg, ModTime: time.Unix(0, 0),
		}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatalf("tar header: %v", err)
		}
		if _, err := tw.Write([]byte(content)); err != nil {
			t.Fatalf("tar write: %v", err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("tar close: %v", err)
	}
	return buf.Bytes()
}

// --- DetectKind ------------------------------------------------------------

func TestDetectKind(t *testing.T) {
	cases := []struct {
		name string
		want ArchiveKind
	}{
		{"foo.tar.gz", KindTarGz},
		{"foo.tgz", KindTarGz},
		{"foo.tar", KindTar},
		{"foo.zip", KindZip},
		{"foo.txt", KindUnknown},
		{"path/to/X.TAR.GZ", KindTarGz},
	}
	for _, tc := range cases {
		if got := DetectKind(tc.name); got != tc.want {
			t.Errorf("%s: want %v got %v", tc.name, tc.want, got)
		}
	}
}

// --- streamArchive (unit) --------------------------------------------------

func TestStreamArchive_TarGz_Flat(t *testing.T) {
	data := makeTarGz(t, map[string]string{"a.xml": xmlOne, "ignored.bin": "x"})
	collectAndCheck(t, data, KindTarGz, 2)
}

func TestStreamArchive_TarGz_Nested(t *testing.T) {
	data := makeNestedTarGz(t, map[string]string{"a.xml": xmlOne})
	collectAndCheck(t, data, KindTarGz, 2)
}

func collectAndCheck(t *testing.T, data []byte, kind ArchiveKind, wantRecords int) {
	t.Helper()
	tmpDir := t.TempDir()
	stream := streamArchive(context.Background(), "test", bytes.NewReader(data), kind, WalkConfig{SpoolDir: tmpDir})

	ext := NewXMLStreamExtractor()
	var got []PatentRecord
	for entry := range stream {
		if entry.Error != nil {
			t.Fatalf("entry error: %v", entry.Error)
		}
		recStream := ext.Stream(context.Background(), entry.Value)
		for r := range recStream {
			if r.Error != nil {
				t.Fatalf("record error: %v", r.Error)
			}
			got = append(got, r.Value)
		}
	}
	if len(got) != wantRecords {
		t.Fatalf("want %d records, got %d", wantRecords, len(got))
	}
	if got[0].PatentID != "EP1000001A1" {
		t.Errorf("want PatentID=EP1000001A1, got %q", got[0].PatentID)
	}
}

// --- End-to-end pipeline test (HTTP source + opener + sink) ----------------

func TestPipeline_EndToEnd_HTTPTarGz(t *testing.T) {
	archive := makeTarGz(t, map[string]string{"docs/a.xml": xmlOne})
	mux := http.NewServeMux()
	// Mimic the EPO product-listing endpoint shape used by EPOProductSource.
	mux.HandleFunc("/products/3", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"id":3,"name":"test","deliveries":[{"deliveryId":1,"deliveryName":"d1","items":[{"itemId":42,"itemName":"a.tar.gz","fileSize":"0","fileChecksum":""}]}]}`)
	})
	mux.HandleFunc("/products/3/delivery/1/item/42/download", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/gzip")
		_, _ = w.Write(archive)
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	src := NewEPOProductSource(srv.URL, 3, srv.Client())
	opener := NewHTTPOpener(srv.Client(), 0, false, WalkConfig{SpoolDir: t.TempDir()})
	sink := &MemorySink{}

	p, err := New(
		WithSource(src),
		WithOpener(opener),
		WithExtractor(NewXMLStreamExtractor()),
		WithSink(sink),
		WithBatchSize(10),
		WithBatchTimeout(100*time.Millisecond),
	)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := p.Run(ctx); err != nil {
		t.Fatalf("run: %v", err)
	}
	if len(sink.Records) != 2 {
		t.Fatalf("want 2 records, got %d", len(sink.Records))
	}
}

// --- Local-dir replay test (LocalFileOpener + StaticListSource) ------------

func TestPipeline_LocalReplay_NoNetwork(t *testing.T) {
	dir := t.TempDir()
	archivePath := filepath.Join(dir, "a.tar.gz")
	if err := os.WriteFile(archivePath, makeTarGz(t, map[string]string{"a.xml": xmlOne}), 0o600); err != nil {
		t.Fatal(err)
	}
	src := &StaticListSource{Jobs: []ArchiveJob{{Name: "a.tar.gz", LocalPath: archivePath}}}
	sink := &MemorySink{}
	p, _ := New(
		WithSource(src),
		WithOpener(NewLocalFileOpener(WalkConfig{SpoolDir: t.TempDir()})),
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
}

// --- Cancellation ----------------------------------------------------------

func TestPipeline_ContextCancellation_BeforeStart(t *testing.T) {
	src := &StaticListSource{Jobs: []ArchiveJob{{Name: "a.tar.gz", LocalPath: "/nonexistent"}}}
	sink := &MemorySink{}
	p, _ := New(
		WithSource(src),
		WithOpener(NewLocalFileOpener(WalkConfig{SpoolDir: t.TempDir()})),
		WithExtractor(NewXMLStreamExtractor()),
		WithSink(sink),
		WithBatchSize(1),
		WithBatchTimeout(10*time.Millisecond),
	)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel before Run
	// Either ctx.Err() or the open-failure error is acceptable; what
	// matters is that Run terminates promptly without leaking goroutines.
	_ = p.Run(ctx)
}

// --- Validation ------------------------------------------------------------

func TestNew_RequiresAllSeams(t *testing.T) {
	_, err := New()
	if err == nil || !strings.Contains(err.Error(), "Source") {
		t.Fatalf("expected Source-required error, got %v", err)
	}
}
