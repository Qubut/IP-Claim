package cmd

import (
	"context"
	"io"
	"os"
	"sync"
	"sync/atomic"

	A "github.com/IBM/fp-go/v2/array"
	IOG "github.com/IBM/fp-go/v2/io"
	"github.com/jedib0t/go-pretty/v6/text"
	"github.com/mattn/go-isatty"
	"github.com/vbauerster/mpb/v8"
	"github.com/vbauerster/mpb/v8/decor"

	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/pipeline"
)

// colorMeta lifts a go-pretty color set into the func(string) string "meta"
// hook used by mpb styles/decorators. mpb measures the un-wrapped string for
// width, then applies this wrapper, so ANSI codes never skew bar alignment.
func colorMeta(c text.Colors) func(string) string {
	return func(s string) string { return c.Sprint(s) }
}

// barStyle is the colored download-bar filler: green fill, bright-green tip.
func barStyle() mpb.BarStyleComposer {
	return mpb.BarStyle().
		Lbound("[").Rbound("]").
		Filler("=").Tip(">").Padding("-").
		FillerMeta(colorMeta(text.Colors{text.FgGreen})).
		TipMeta(colorMeta(text.Colors{text.FgHiGreen}))
}

// progressBar is the minimal single-bar surface used by process-hupd and
// analyze (which track a single stream rather than many archives). All insight
// lives in the Describe message; the spinner animates via auto-refresh.
type progressBar interface {
	Describe(string)
	Finish() error
}

// noopProgressBar is the silent fallback used when stderr is not a TTY.
type noopProgressBar struct{}

func (noopProgressBar) Describe(string) {}
func (noopProgressBar) Finish() error   { return nil }

// teardown is one bar's deferred abort captured as an effect value (fp-go
// IO). The display keeps a list of these rather than scattering bars across
// typed fields, so stop() simply runs every finalizer and can never miss a
// bar — the invariant the previous hand-rolled cleanup violated for single
// spinners (leaving them to hang p.Wait until ^C).
type teardown = IOG.IO[any]

// display owns the mpb multi-bar container for a command run. It renders to
// stderr (stdout stays clean for the summary table) and is a no-op when
// stderr is not a TTY, so piped/redirected runs emit no control codes.
//
// Layout for `process`: one persistent aggregate spinner showing rolled-up
// pipeline stats, plus a transient per-archive download bar created on first
// byte and dropped when the archive settles. `process-hupd`/`analyze` use the
// single-bar adapter instead.
type display struct {
	p *mpb.Progress // nil when not a TTY

	aggMsg atomic.Pointer[string] // aggregate description, read during render

	mu      sync.Mutex
	bars    map[string]*mpb.Bar // live per-archive bars, keyed for update/drop
	closers []teardown          // one finalizer per created bar; drives stop()
}

// track registers bar's teardown and returns the bar. Callers MUST hold d.mu.
// Routing every bar through this single choke point is what makes stop()
// exhaustive: a bar cannot be created without also being scheduled for abort.
func (d *display) track(bar *mpb.Bar) *mpb.Bar {
	d.closers = append(d.closers, func() any { bar.Abort(true); return nil })
	return bar
}

// newDisplay builds a display. When stderr is not a TTY the container is left
// nil and every method degrades to a no-op.
func newDisplay(ctx context.Context) *display {
	d := &display{bars: map[string]*mpb.Bar{}}
	if isatty.IsTerminal(os.Stderr.Fd()) {
		d.p = mpb.NewWithContext(ctx,
			mpb.WithOutput(os.Stderr),
			mpb.WithAutoRefresh(), // refresh on a timer, not only on bar updates
		)
	}
	return d
}

// active reports whether a live container is rendering.
func (d *display) active() bool { return d != nil && d.p != nil }

// writer returns the io.Writer that interleaves log lines above the bars, or
// nil when inactive. mpb.Progress.Write redraws the bars after each line.
func (d *display) writer() io.Writer {
	if !d.active() {
		return nil
	}
	return d.p
}

// setAggregate updates the aggregate spinner's message.
func (d *display) setAggregate(msg string) { d.aggMsg.Store(&msg) }

func (d *display) loadAggregate() string {
	if s := d.aggMsg.Load(); s != nil {
		return *s
	}
	return ""
}

// startAggregate adds the persistent aggregate spinner. Safe to call when
// inactive.
func (d *display) startAggregate(label string) {
	if !d.active() {
		return
	}
	d.setAggregate(label)
	d.mu.Lock()
	defer d.mu.Unlock()
	d.track(d.p.New(0,
		mpb.SpinnerStyle().Meta(colorMeta(text.Colors{text.FgHiCyan})),
		mpb.PrependDecorators(decor.Meta(
			decor.Any(func(decor.Statistics) string { return d.loadAggregate() }),
			colorMeta(text.Colors{text.FgHiWhite, text.Bold}),
		)),
		mpb.AppendDecorators(decor.Meta(
			decor.Elapsed(decor.ET_STYLE_MMSS),
			colorMeta(text.Colors{text.FgHiBlack}),
		)),
	))
}

// onBytes feeds byte-level download progress, lazily creating a per-archive
// bar on first sight. Safe for concurrent archives. No-op when inactive.
func (d *display) onBytes(job pipeline.ArchiveJob, downloaded, total int64) {
	if !d.active() {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	bar, ok := d.bars[job.Name]
	if !ok {
		bar = d.track(d.p.New(total,
			barStyle(),
			mpb.BarRemoveOnComplete(),
			mpb.PrependDecorators(decor.Meta(
				decor.Name(job.Name, decor.WCSyncSpaceR),
				colorMeta(text.Colors{text.FgHiCyan}),
			)),
			mpb.AppendDecorators(
				decor.CountersKibiByte("% .1f / % .1f", decor.WCSyncWidth),
				decor.Meta(
					decor.Percentage(decor.WCSyncSpace),
					colorMeta(text.Colors{text.FgHiYellow}),
				),
			),
		))
		d.bars[job.Name] = bar
	}
	if total > 0 {
		bar.SetTotal(total, false)
	}
	bar.SetCurrent(downloaded)
}

// onSettled drops the per-archive bar once the archive terminally succeeds or
// fails. No-op when inactive or unknown.
func (d *display) onSettled(job pipeline.ArchiveJob, _ error) {
	if !d.active() {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if bar, ok := d.bars[job.Name]; ok {
		bar.Abort(true) // drop without trace; aggregate already reflects totals
		delete(d.bars, job.Name)
	}
}

// stop aborts every tracked bar and blocks until the container has drained.
// Safe to call when inactive and idempotent across repeated calls.
//
// Teardown is exhaustive by construction: it runs the finalizer registry built
// by track, so it cannot miss a bar kind (a spinner with total 0 never
// auto-completes, so an un-aborted one would block p.Wait forever).
func (d *display) stop() {
	if !d.active() {
		return
	}
	d.mu.Lock()
	closers := d.closers
	d.closers = nil
	d.bars = map[string]*mpb.Bar{}
	p := d.p
	d.p = nil // idempotent: subsequent stop()/active() calls are no-ops
	d.mu.Unlock()

	// Run every captured abort effect, then let the container drain.
	A.Reduce(func(_ any, t teardown) any { return t() }, any(nil))(closers)
	p.Wait()
}

// singleBar returns a progressBar driving one mpb spinner (label + live
// counter). When inactive it returns noopProgressBar so callers stay
// nil-free.
func (d *display) singleBar(label string) progressBar {
	if !d.active() {
		return noopProgressBar{}
	}
	sb := &mpbSingleBar{}
	sb.desc.Store(&label)
	d.mu.Lock()
	defer d.mu.Unlock()
	sb.bar = d.track(d.p.New(0,
		mpb.SpinnerStyle().Meta(colorMeta(text.Colors{text.FgHiCyan})),
		mpb.PrependDecorators(decor.Meta(
			decor.Any(func(decor.Statistics) string { return sb.load() }),
			colorMeta(text.Colors{text.FgHiWhite, text.Bold}),
		)),
		mpb.AppendDecorators(decor.Meta(
			decor.Elapsed(decor.ET_STYLE_MMSS),
			colorMeta(text.Colors{text.FgHiBlack}),
		)),
	))
	return sb
}

// mpbSingleBar adapts a single mpb spinner to the progressBar interface.
type mpbSingleBar struct {
	bar  *mpb.Bar
	desc atomic.Pointer[string]
}

func (s *mpbSingleBar) load() string {
	if d := s.desc.Load(); d != nil {
		return *d
	}
	return ""
}

func (s *mpbSingleBar) Describe(msg string) { s.desc.Store(&msg) }
func (s *mpbSingleBar) Finish() error       { s.bar.Abort(true); return nil }
