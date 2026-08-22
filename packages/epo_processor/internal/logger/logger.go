package logger

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	A "github.com/IBM/fp-go/v2/array"
	E "github.com/IBM/fp-go/v2/either"
	F "github.com/IBM/fp-go/v2/function"
	IOR "github.com/IBM/fp-go/v2/idiomatic/ioresult"
	O "github.com/IBM/fp-go/v2/option"
	P "github.com/IBM/fp-go/v2/predicate"
	"github.com/lmittmann/tint"
	"github.com/mattn/go-isatty"
)

// firstError runs f over every element of xs (effects always fire, fan-out
// style) and returns the first non-nil error
func firstError[T any](xs []T, f func(T) error) error {
	return F.Pipe1(xs, A.Reduce(func(acc error, x T) error {
		e := f(x)
		return F.Pipe2(acc, O.FromPredicate(P.IsNonZero[error]()), O.GetOrElse(F.Constant(e)))
	}, error(nil)))
}

// Controls exposes the runtime-steerable pieces of the logger 
type Controls struct {
	// Console steers the console handler's writer and level.
	Console *ConsoleControl
	// Errors accumulates every record at WARN or above.
	Errors *Collector
	// ErrorsPath is the on-disk errors log (empty when no log dir).
	ErrorsPath string
}

// New builds a *slog.Logger that writes human-readable text to stderr
func New(logLevel, logDir string) (*slog.Logger, *Controls, func() error, error) {
	base := parseLevel(logLevel)

	consoleW := newSwWriter(os.Stderr)
	lvl := new(slog.LevelVar)
	lvl.Set(base)

	// Color the console only when stderr is a real terminal;
	consoleColor := isatty.IsTerminal(os.Stderr.Fd()) || isatty.IsCygwinTerminal(os.Stderr.Fd())

	collector := &Collector{}
	handlers := []slog.Handler{
		tint.NewHandler(consoleW, &tint.Options{
			Level:      lvl,
			TimeFormat: "15:04:05",
			NoColor:    !consoleColor,
		}),
		collector,
	}

	closer := noopCloser
	errorsPath := ""
	if logDir != "" {
		ts := time.Now().Format("20060102-150405")
		mainPath := filepath.Join(logDir, fmt.Sprintf("epo-processor[%s].log", ts))
		errorsPath = filepath.Join(logDir, fmt.Sprintf("epo-processor[%s].errors.jsonl", ts))

		// Open both JSON sinks as one IOResult
		sinks, err := IOR.SequenceT2(
			openSink(base)(mainPath),
			openSink(slog.LevelWarn)(errorsPath),
		)()
		if err != nil {
			return nil, nil, nil, err
		}
		handlers = append(handlers, sinks.F1.handler, sinks.F2.handler)
		closer = closeAll(sinks.F1.closer, sinks.F2.closer)
	}

	ctl := &Controls{
		Console:    &ConsoleControl{w: consoleW, level: lvl, base: base},
		Errors:     collector,
		ErrorsPath: errorsPath,
	}
	return slog.New(newMultiHandler(handlers...)), ctl, closer, nil
}

// fileSink pairs a file-backed JSON handler with the closer that releases it.
type fileSink struct {
	handler slog.Handler
	closer  func() error
}

// openAppend opens (creating if needed) a file for append-only writes.
func openAppend(path string) (*os.File, error) {
	return os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600) //nolint:gosec // path from config
}

// openAppendIOR lifts openAppend into the ioresult monad.
var openAppendIOR = IOR.Eitherize1(openAppend)

// openSink is the Kleisli arrow path -> IOResult[fileSink]: open the file for
// append and wrap it in a JSON handler emitting records at or above level.
func openSink(level slog.Level) IOR.Kleisli[string, fileSink] {
	return F.Flow2(
		openAppendIOR,
		IOR.Map(func(f *os.File) fileSink {
			return fileSink{
				handler: slog.NewJSONHandler(f, &slog.HandlerOptions{Level: level}),
				closer:  f.Close,
			}
		}),
	)
}

// noopCloser is the closer used when no files were opened.
func noopCloser() error { return nil }

// closeAll returns a closer that runs every cs and yields the first error.
func closeAll(cs ...func() error) func() error {
	return func() error {
		return firstError(cs, func(c func() error) error { return c() })
	}
}

// parseLevel maps a textual level to slog.Level, defaulting to info.
func parseLevel(s string) slog.Level {
	var l slog.Level
	err := l.UnmarshalText([]byte(strings.TrimSpace(s)))
	return F.Pipe1(
		E.TryCatchError(l, err),
		E.GetOrElse(F.Constant1[error](slog.LevelInfo)),
	)
}

// ConsoleControl steers the console handler at runtime.
type ConsoleControl struct {
	w     *swWriter
	level *slog.LevelVar
	base  slog.Level
}

// SetTTYMode repoints console output at w (the live progress display) and, if
// the configured level is below WARN, raises it to WARN for the duration.
func (c *ConsoleControl) SetTTYMode(w io.Writer) {
	c.w.Set(w)
	if c.base < slog.LevelWarn {
		c.level.Set(slog.LevelWarn)
	}
}

// Restore reverts the console writer to stderr and the level to its config value.
func (c *ConsoleControl) Restore() {
	c.w.Reset()
	c.level.Set(c.base)
}

// swWriter is an io.Writer whose target can be swapped atomically. The console
// slog handler is built once over it; SetTTYMode/Restore repoint it without
// rebuilding handlers.
type swWriter struct {
	w   atomic.Pointer[io.Writer]
	def io.Writer
}

func newSwWriter(def io.Writer) *swWriter { return &swWriter{def: def} }

func (s *swWriter) Write(p []byte) (int, error) {
	if w := s.w.Load(); w != nil {
		return (*w).Write(p)
	}
	return s.def.Write(p)
}

func (s *swWriter) Set(w io.Writer) { s.w.Store(&w) }
func (s *swWriter) Reset()          { s.w.Store(nil) }

// Entry is a single recorded warning/error.
type Entry struct {
	Time    time.Time
	Level   slog.Level
	Message string
	Archive string // from the "name"/"label" attr when present
}

// Collector records every WARN+ record in memory and tallies counts.
type Collector struct {
	mu      sync.Mutex
	entries []Entry
	warn    int
	errc    int
}

func (c *Collector) Enabled(_ context.Context, l slog.Level) bool { return l >= slog.LevelWarn }

func (c *Collector) Handle(_ context.Context, r slog.Record) error {
	e := Entry{Time: r.Time, Level: r.Level, Message: r.Message}
	r.Attrs(func(a slog.Attr) bool {
		if e.Archive == "" && (a.Key == "name" || a.Key == "label") {
			e.Archive = a.Value.String()
		}
		return true
	})
	c.mu.Lock()
	defer c.mu.Unlock()
	c.entries = append(c.entries, e)
	if r.Level >= slog.LevelError {
		c.errc++
	} else {
		c.warn++
	}
	return nil
}

func (c *Collector) WithAttrs([]slog.Attr) slog.Handler { return c }
func (c *Collector) WithGroup(string) slog.Handler      { return c }

// Counts returns the number of recorded warnings and errors.
func (c *Collector) Counts() (warn, errc int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.warn, c.errc
}

// Entries returns a copy of all recorded WARN+ entries.
func (c *Collector) Entries() []Entry {
	c.mu.Lock()
	defer c.mu.Unlock()
	return slices.Clone(c.entries)
}

// Archives returns the distinct, non-empty archive names that produced a
// warning or error.
func (c *Collector) Archives() []string {
	return F.Pipe3(
		c.Entries(),
		A.Map(func(e Entry) string { return e.Archive }),
		A.Filter(P.IsNonZero[string]()),
		A.StrictUniq[string],
	)
}

type multiHandler struct{ handlers []slog.Handler }

func newMultiHandler(hs ...slog.Handler) *multiHandler { return &multiHandler{handlers: hs} }

func (m *multiHandler) Enabled(ctx context.Context, l slog.Level) bool {
	return F.Pipe1(m.handlers, A.Reduce(func(ok bool, h slog.Handler) bool {
		return ok || h.Enabled(ctx, l)
	}, false))
}

// Handle fans the record out to every enabled handler (each gets its own clone) and returns the first error.
func (m *multiHandler) Handle(ctx context.Context, r slog.Record) error {
	return firstError(m.handlers, func(h slog.Handler) error {
		return F.Pipe2(
			h,
			O.FromPredicate(func(h slog.Handler) bool { return h.Enabled(ctx, r.Level) }),
			O.Fold(
				F.Constant[error](nil),
				func(h slog.Handler) error { return h.Handle(ctx, r.Clone()) },
			),
		)
	})
}

func (m *multiHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return &multiHandler{handlers: F.Pipe1(m.handlers, A.Map(func(h slog.Handler) slog.Handler {
		return h.WithAttrs(attrs)
	}))}
}

func (m *multiHandler) WithGroup(name string) slog.Handler {
	return &multiHandler{handlers: F.Pipe1(m.handlers, A.Map(func(h slog.Handler) slog.Handler {
		return h.WithGroup(name)
	}))}
}
