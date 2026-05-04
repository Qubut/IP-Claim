package pipeline

import (
	"context"
	"crypto/sha1" //nolint:gosec // EPO BDDS mandates SHA-1 checksums; not a security primitive here
	"encoding/hex"
	"fmt"
	"hash"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	IOR "github.com/IBM/fp-go/v2/idiomatic/ioresult"
	"github.com/IBM/fp-go/v2/retry"
	"github.com/destel/rill"
	"go.uber.org/zap"
)

// HTTPOpener fetches each archive over HTTP and feeds it to the shared
// archive walker. Tar/tar.gz are processed without touching disk; zip is
// spooled to Walk.SpoolDir. Retry, optional SHA-1 verification, and
// resource cleanup are layered via fp-go ioresult Bracket+Retrying.
type HTTPOpener struct {
	Client     *http.Client
	MaxRetries uint
	VerifySHA1 bool
	Walk       WalkConfig

	// KeepArchive, when true, tees the HTTP response body to a file under
	// ArchiveDir while the walker consumes it. Orthogonal to
	// Walk.KeepExtracted.
	KeepArchive bool
	ArchiveDir  string

	// Logger, when non-nil, receives structured warnings (e.g. for
	// unsupported archive kinds).
	Logger *zap.SugaredLogger

	// Janitor receives a per-archive completion notification after the
	// stream is fully drained without error. Defaults to NoopJanitor.
	Janitor Janitor

	// OnBytes, when non-nil, receives throttled byte-level progress for
	// the current download (downloaded, total). total is -1 when the
	// server omits Content-Length. Called from the producer goroutine;
	// must not block.
	OnBytes func(job ArchiveJob, downloaded, total int64)

	// ProgressInterval throttles OnBytes calls. Defaults to 100ms.
	ProgressInterval time.Duration
}

// NewHTTPOpener returns an HTTPOpener with sensible defaults. Set
// KeepArchive and ArchiveDir on the result to enable archive-level
// retention.
func NewHTTPOpener(client *http.Client, maxRetries uint, verifySHA1 bool, walk WalkConfig) *HTTPOpener {
	if client == nil {
		client = &http.Client{Timeout: 0}
	}
	walk.SpoolDir = resolvedSpoolDir(walk.SpoolDir)
	return &HTTPOpener{
		Client:     client,
		MaxRetries: maxRetries,
		VerifySHA1: verifySHA1,
		Walk:       walk,
		Janitor:    NoopJanitor{},
	}
}

// Stream issues the GET, runs the archive walk under Bracket+Retrying,
// and notifies the Janitor on success.
func (o *HTTPOpener) Stream(ctx context.Context, job ArchiveJob) rill.Stream[XMLEntry] {
	return rill.Generate(func(send func(XMLEntry), sendErr func(error)) {
		// Unsupported archive kinds (e.g. .csv siblings) yield empty
		// streams; sibling archives keep flowing.
		if DetectKind(job.Name) == KindUnknown {
			whenLog(o.Logger, func(l *zap.SugaredLogger) {
				l.Warnw("archive: skipping unsupported kind", "name", job.Name)
			})
			return
		}
		policy := retry.Monoid.Concat(
			retry.LimitRetries(o.MaxRetries),
			retry.ExponentialBackoff(50*time.Millisecond),
		)
		fetch := func(_ retry.RetryStatus) IOR.IOResult[struct{}] {
			return IOR.Bracket(
				o.acquire(ctx, job),
				func(resp *http.Response) IOR.IOResult[struct{}] {
					return o.consume(ctx, job, resp, send, sendErr)
				},
				releaseResponse,
			)
		}
		check := func(_ struct{}, err error) bool {
			return err != nil && ctx.Err() == nil
		}
		_, err := IOR.Retrying(policy, fetch, check)()
		if err != nil {
			sendErr(err)
			return
		}
		o.Janitor.OnArchiveDone(ctx, job, "", o.KeepArchive)
	})
}

// acquire issues the GET and returns the *http.Response on a 200, or an
// error otherwise. The body is owned by the use-callback under Bracket.
func (o *HTTPOpener) acquire(ctx context.Context, job ArchiveJob) IOR.IOResult[*http.Response] {
	return func() (*http.Response, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, job.URL, nil)
		if err != nil {
			return nil, err
		}
		resp, err := o.Client.Do(req)
		if err != nil {
			return nil, err
		}
		if resp.StatusCode != http.StatusOK {
			_ = resp.Body.Close()
			return nil, fmt.Errorf("download %s: bad status %d", job.URL, resp.StatusCode)
		}
		return resp, nil
	}
}

// consume reads the body through an optional SHA-1 tee and an optional
// kept-archive tee, feeds the archive walker, and verifies the checksum
// (if any). On checksum mismatch returns an error so Retrying can re-run.
func (o *HTTPOpener) consume(
	ctx context.Context,
	job ArchiveJob,
	resp *http.Response,
	send func(XMLEntry),
	sendErr func(error),
) IOR.IOResult[struct{}] {
	return func() (struct{}, error) {
		var (
			bodyReader io.Reader = resp.Body
			h          hash.Hash
		)
		if o.OnBytes != nil {
			bodyReader = newProgressReader(bodyReader, resp.ContentLength, o.ProgressInterval,
				func(read, total int64) { o.OnBytes(job, read, total) })
		}
		if o.VerifySHA1 && job.Checksum != "" {
			h = sha1.New() //nolint:gosec // EPO BDDS mandates SHA-1 checksums; not used for security
			bodyReader = io.TeeReader(bodyReader, h)
		}
		bodyReader, closeArchive, err := optionalTee(o.KeepArchive, "", job.Name, o.ArchiveDir, bodyReader)
		if err != nil {
			return struct{}{}, err
		}
		defer func() { _ = closeArchive() }()

		kind := DetectKind(job.Name)
		if err := walkArchive(ctx, job.Name, bodyReader, kind, o.Walk, send, sendErr); err != nil {
			return struct{}{}, err
		}
		// Drain trailing bytes so the SHA-1 / kept-archive tee covers the
		// full stream even if the walker exited early.
		_, _ = io.Copy(io.Discard, bodyReader)
		if h != nil {
			// Hex digests are case-insensitive (RFC 4648 §8).
			if actual := hex.EncodeToString(h.Sum(nil)); !strings.EqualFold(actual, job.Checksum) {
				return struct{}{}, fmt.Errorf(
					"checksum mismatch for %s: want %s got %s",
					job.Name, job.Checksum, actual,
				)
			}
		}
		return struct{}{}, nil
	}
}

// releaseResponse closes the HTTP response body. Curried per the
// [IOR.Bracket] release-callback shape.
func releaseResponse(_ struct{}, _ error) func(*http.Response) IOR.IOResult[any] {
	return func(resp *http.Response) IOR.IOResult[any] {
		return func() (any, error) {
			if resp != nil && resp.Body != nil {
				return nil, resp.Body.Close()
			}
			return nil, nil
		}
	}
}

// LocalFileOpener streams archives from the local filesystem using the
// same walker as HTTPOpener. Useful for re-runs without re-downloading.
type LocalFileOpener struct {
	Walk    WalkConfig
	Logger  *zap.SugaredLogger
	Janitor Janitor
}

// NewLocalFileOpener creates a LocalFileOpener that reads archives from local paths.
func NewLocalFileOpener(walk WalkConfig) *LocalFileOpener {
	walk.SpoolDir = resolvedSpoolDir(walk.SpoolDir)
	return &LocalFileOpener{Walk: walk, Janitor: NoopJanitor{}}
}

// Stream opens the local file specified by job and emits its XML entries.
func (l *LocalFileOpener) Stream(ctx context.Context, job ArchiveJob) rill.Stream[XMLEntry] {
	return rill.Generate(func(send func(XMLEntry), sendErr func(error)) {
		path := job.LocalPath
		if path == "" {
			path = job.Name
		}
		if DetectKind(path) == KindUnknown {
			whenLog(l.Logger, func(lg *zap.SugaredLogger) {
				lg.Warnw("archive: skipping unsupported kind", "name", job.Name)
			})
			return
		}
		f, err := os.Open(path) //nolint:gosec // path from user config, not tainted
		if err != nil {
			sendErr(fmt.Errorf("open %s: %w", path, err))
			return
		}
		defer func() { _ = f.Close() }()
		if err := walkArchive(ctx, job.Name, f, DetectKind(path), l.Walk, send, sendErr); err != nil {
			sendErr(err)
			return
		}
		l.Janitor.OnArchiveDone(ctx, job, path, true)
	})
}
