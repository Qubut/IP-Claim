package cmd

import (
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/jedib0t/go-pretty/v6/table"
	"github.com/jedib0t/go-pretty/v6/text"
	"github.com/mattn/go-isatty"

	applog "github.com/Qubut/IP-Claim/packages/epo_processor/internal/logger"
)

// kv is a single metric row in a summary table.
type kv struct {
	key string
	val string
}

// renderSummary prints a compact, titled two-column summary table to stdout.
func renderSummary(title string, rows []kv) {
	renderTable(title, rows, text.Colors{text.FgHiCyan, text.Bold})
}

// renderTable renders a titled two-column key/value table. Colors are applied
// only when stdout is a real terminal so piped output stays plain.
func renderTable(title string, rows []kv, titleColors text.Colors) {
	color := isatty.IsTerminal(os.Stdout.Fd()) || isatty.IsCygwinTerminal(os.Stdout.Fd())

	t := table.NewWriter()
	t.SetOutputMirror(os.Stdout)
	t.SetTitle(title)
	t.SetStyle(table.StyleRounded)

	keyCol := table.ColumnConfig{Number: 1, WidthMax: 24}
	// Cap the value column so a long field (e.g. a list of archives) wraps
	// instead of stretching the table across the whole terminal.
	valCol := table.ColumnConfig{Number: 2, WidthMax: 72, WidthMaxEnforcer: text.WrapSoft}
	if color {
		t.Style().Title.Colors = titleColors
		t.Style().Color.Border = text.Colors{text.FgHiBlack}
		t.Style().Color.Separator = text.Colors{text.FgHiBlack}
		keyCol.Colors = text.Colors{text.FgCyan}
	}
	t.SetColumnConfigs([]table.ColumnConfig{keyCol, valCol})

	for _, r := range rows {
		t.AppendRow(table.Row{r.key, r.val})
	}
	t.Render()
}

// renderIssues prints an end-of-run issues table when the run recorded any
// warnings or errors. The full per-event detail lives in the errors log file.
func renderIssues(ctl *applog.Controls) {
	if ctl == nil || ctl.Errors == nil {
		return
	}
	warn, errc := ctl.Errors.Counts()
	if warn == 0 && errc == 0 {
		return
	}
	rows := []kv{
		{"Warnings", strconv.Itoa(warn)},
		{"Errors", strconv.Itoa(errc)},
	}
	if affected := ctl.Errors.Archives(); len(affected) > 0 {
		rows = append(rows, kv{"Affected archives", joinCapped(affected, 10)})
	}
	if ctl.ErrorsPath != "" {
		rows = append(rows, kv{"Errors log", ctl.ErrorsPath})
	}

	renderTable("Issues: warnings & errors", rows, text.Colors{text.FgHiYellow, text.Bold})
}

// joinCapped joins up to max names with ", "; any remainder is summarised as a
// trailing "(+N more)" so the issues table never lists hundreds of entries.
func joinCapped(items []string, max int) string {
	if len(items) <= max {
		return strings.Join(items, ", ")
	}
	return strings.Join(items[:max], ", ") +
		", (+" + strconv.Itoa(len(items)-max) + " more)"
}

// ratePerSec returns n/elapsed as a per-second rate, guarding against a
// zero or negative elapsed window.
func ratePerSec(n int64, elapsed time.Duration) float64 {
	secs := elapsed.Seconds()
	if secs <= 0 {
		return 0
	}
	return float64(n) / secs
}
