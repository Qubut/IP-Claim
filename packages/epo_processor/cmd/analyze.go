package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	A "github.com/IBM/fp-go/v2/array"
	F "github.com/IBM/fp-go/v2/function"
	IOResF "github.com/IBM/fp-go/v2/idiomatic/ioresult/file"
	P "github.com/IBM/fp-go/v2/predicate"
	"github.com/destel/rill"
	"github.com/parquet-go/parquet-go"
	"github.com/spf13/cobra"

	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/hupd"
	"github.com/Qubut/IP-Claim/packages/epo_processor/internal/parse"
)

// analyzeCmd reports overlap between an EPO record set and a HUPD
// extraction directory. EPO can be Parquet or CSV (chosen by extension).
//
// Memory model: the only large structure held resident is the HUPD ID
// set (one entry per HUPD JSON file). The EPO file is streamed; rows
// are inspected and discarded. Collision tracking is opt-in via
// --collisions because it requires a second EPO pass and a per-family
// counters map proportional to the EPO family graph.
var analyzeCmd = &cobra.Command{
	Use:   "analyze <hupd-dir> [epo-file]",
	Short: "Build a linked EPO↔HUPD Parquet dataset and report citation overlap",
	Long: "Report overlap between an EPO record set and an extracted HUPD directory.\n\n" +
		"Positional args (order-independent):\n" +
		"  <hupd-dir>   root of extracted HUPD JSON files (recursive); the directory arg - required\n" +
		"  [epo-file]   EPO records (.parquet/.csv/.csv.gz); the file arg; defaults to pipeline.output_parquet\n\n" +
		"The directory arg is taken as the HUPD dir and the file arg as the EPO file,\n" +
		"so the two may be given in either order. With one arg it is the HUPD dir.",
	Args: cobra.RangeArgs(1, 2),
	RunE: func(cmd *cobra.Command, args []string) error {
		ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer cancel()

		// Positional args are order-independent: the HUPD source is a
		// directory, the EPO record set is a file (.parquet/.csv/.csv.gz). We
		// classify by filesystem type so `analyze <hupd-dir> <epo-file>` and
		// `analyze <epo-file> <hupd-dir>` both work. With one arg it is the
		// HUPD dir (EPO defaults to pipeline.output_parquet).
		var epoPath, hupdDir string
		switch {
		case len(args) == 1:
			hupdDir = args[0]
		case isDir(args[0]) && !isDir(args[1]):
			hupdDir, epoPath = args[0], args[1]
		case isDir(args[1]) && !isDir(args[0]):
			hupdDir, epoPath = args[1], args[0]
		default:
			// Ambiguous (both dirs, both files, or neither exists): fall back
			// to the documented [epo-file] <hupd-dir> order.
			epoPath, hupdDir = args[0], args[1]
		}
		metaPath, _ := cmd.Flags().GetString("meta")
		outputJSON, _ := cmd.Flags().GetString("out")
		datasetPath, _ := cmd.Flags().GetString("dataset")
		minCollisions, _ := cmd.Flags().GetInt("min-collisions")
		workers, _ := cmd.Flags().GetInt("workers")
		if workers <= 0 {
			workers = runtime.NumCPU()
		}

		if strings.TrimSpace(epoPath) == "" {
			epoPath = cfg.Pipeline.OutputParquet
		}
		if strings.TrimSpace(epoPath) == "" {
			return fmt.Errorf("EPO file is required: pass it as the first positional arg or set pipeline.output_parquet")
		}
		if strings.TrimSpace(hupdDir) == "" {
			return fmt.Errorf("HUPD dir is required (last positional arg)")
		}
		if minCollisions < 1 {
			minCollisions = 1
		}

		disp := newDisplay(ctx)
		tty := disp.active()
		if tty {
			logCtl.Console.SetTTYMode(disp.writer())
			defer logCtl.Console.Restore()
		}
		bar := disp.singleBar("analyze")
		defer disp.stop()
		var barMu sync.Mutex

		hupdStart := time.Now()
		logger.Info("analyze: scanning HUPD", "dir", hupdDir, "meta", metaPath, "workers", workers)
		var (
			hupdFiles map[string][]string
			err       error
		)
		if strings.TrimSpace(metaPath) != "" {
			hupdFiles, err = hupd.ScanMeta(ctx, metaPath, cfg.Analyze.HUPDMetaURL, hupdDir, bar, &barMu, logger)
		} else {
			hupdFiles, err = hupd.ScanIDs(ctx, hupdDir, workers, bar, &barMu, logger)
		}
		if err != nil {
			return fmt.Errorf("scan HUPD: %w", err)
		}
		logger.Info("analyze: HUPD scanned",
			"ids", len(hupdFiles),
			"elapsed", time.Since(hupdStart).String(),
		)

		epoStart := time.Now()
		logger.Info("analyze: streaming EPO (single pass)", "path", epoPath, "dataset", datasetPath)
		ov, datasetRows, err := streamEPOOnce(ctx, epoPath, hupdFiles, datasetPath, minCollisions, bar, &barMu)
		if err != nil {
			return fmt.Errorf("stream EPO: %w", err)
		}
		logger.Info("analyze: EPO pass done",
			"records", ov.records,
			"direct", ov.direct,
			"family_only", ov.familyOnly,
			"dataset_rows", datasetRows,
			"elapsed", time.Since(epoStart).String(),
		)

		report := newReport(epoPath, hupdDir, len(hupdFiles), ov)
		if datasetPath != "" {
			report.DatasetRows = datasetRows
			report.DatasetFile = datasetPath
		}

		if outputJSON != "" {
			if err := writeJSON(outputJSON, report); err != nil {
				return err
			}
			logger.Info("analyze: report written", "path", outputJSON)
		}

		disp.stop()
		if tty {
			logCtl.Console.Restore()
		}
		printReport(report)
		renderIssues(logCtl)
		return nil
	},
}

// isDir reports whether path exists and is a directory.
func isDir(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

func init() {
	f := analyzeCmd.Flags()
	f.StringP("meta", "m", "", "Pre-downloaded HUPD Feather metadata file. If absent, falls back to full JSON dir scan.")
	f.StringP("out", "o", "", "Optional path to write the full report as JSON.")
	f.String("dataset", "", "Path to write a Parquet dataset (one DatasetRecord per EPO patent with ≥ min-collisions cited-HUPD patents with category annotations).")
	f.Int("min-collisions", 1, "Min cited-HUPD-with-categories count per EPO patent to emit a TrainingRecord (default 1).")
	f.IntP("workers", "w", 0, "Concurrent HUPD JSON readers (0 = NumCPU).")
}

// epoStream opens path as a RecordSource and pushes each PatentRecord as a
// rill.Try value onto the returned channel. Each record is a value copy;
// the source is closed by the goroutine when exhausted or on error.
// Implements the Producer pattern, decoupling record sourcing from processing.
func epoStream(ctx context.Context, path string) (<-chan rill.Try[parse.PatentRecord], error) {
	src, err := parse.OpenRecordSource(path)
	if err != nil {
		return nil, err
	}
	out := make(chan rill.Try[parse.PatentRecord], 256)
	go func() {
		defer close(out)
		defer func() { _ = src.Close() }()
		var rec parse.PatentRecord
		for {
			if ctx.Err() != nil {
				out <- rill.Wrap(parse.PatentRecord{}, ctx.Err())
				return
			}
			ok, err := src.Next(&rec)
			if err != nil {
				out <- rill.Wrap(parse.PatentRecord{}, err)
				return
			}
			if !ok {
				return
			}
			out <- rill.Wrap(rec, nil) // value copy; slices freshly allocated per Next call
		}
	}()
	return out, nil
}

// --- EPO single pass: overlap + optional dataset --------------------------

// streamEPOOnce streams the EPO file exactly once, computing overlap stats
// and — when outPath is non-empty — writing a Parquet dataset in the same pass.
// This halves EPO I/O compared to running overlap and dataset as separate passes.
//
// Overlap stats (always computed):
//   - direct: HUPD IDs that match the EPO patent_id directly
//   - familyOnly: HUPD IDs reached only via the EPO family graph
//
// Dataset rows (only when outPath != ""):
//   - One DatasetRecord per EPO patent with ≥ minCollisions cited-HUPD patents.
//     ≥ minCollisions cited-HUPD patents with category annotations.
//
// Patterns: Value Object, Repository, Filter-Map (fp-go), Bracket
// (IOResF.Write), Streaming Producer-Consumer.
func streamEPOOnce(
	ctx context.Context,
	epoPath string,
	hupdFiles map[string][]string,
	outPath string, // "" → overlap stats only, no file written
	minCollisions int,
	bar progressBar,
	mu *sync.Mutex,
) (overlapStats, int64, error) {
	if outPath != "" {
		if dir := filepath.Dir(outPath); dir != "" && dir != "." {
			if err := os.MkdirAll(dir, 0o750); err != nil {
				return overlapStats{}, 0, err
			}
		}
	}
	recs, err := epoStream(ctx, epoPath)
	if err != nil {
		return overlapStats{}, 0, err
	}

	inHUPD := func(id string) bool { _, ok := hupdFiles[id]; return ok }

	// Predicate composition (fp-go):
	//   isHUPDCitation = hasCategories ∧ citedInHUPD
	hasCategories := func(c parse.Citation) bool { return len(c.Categories) > 0 }
	citedInHUPD := P.ContraMap(func(c parse.Citation) string { return hupd.NormalizeUSID(c.CitedID) })(inHUPD)
	isHUPDCitation := P.And(hasCategories)(citedInHUPD)

	// Named mappers — defined once, reused in every eligible record.
	toCitedCollision := func(c parse.Citation) CollisionCitation {
		norm := hupd.NormalizeUSID(c.CitedID)
		return CollisionCitation{CitedID: norm, Categories: uniqueUpper(c.Categories), Paths: append([]string(nil), hupdFiles[norm]...)}
	}
	toMember := func(id string) HUPDMember {
		return HUPDMember{ID: id, Paths: append([]string(nil), hupdFiles[id]...)}
	}

	matched := make(map[string]matchKind, 1024)
	var records, rows int64
	var lastLogged, lastBar time.Time
	start := time.Now()

	mu.Lock()
	bar.Describe("EPO↔HUPD")
	mu.Unlock()

	// doPass streams EPO records. w is nil for overlap-only (no dataset).
	doPass := func(w *parquet.GenericWriter[DatasetRecord]) error {
		return rill.ForEach(recs, 1, func(rec parse.PatentRecord) error {
			records++
			epoNorm := hupd.NormalizeUSID(rec.PatentID)
			familyNorm := hupd.NormalizeFamily(rec.FamilyPatents)

			// --- Overlap accounting (always) ---
			if inHUPD(epoNorm) {
				matched[epoNorm] = matchDirect
			}
			matched = A.Reduce(func(acc map[string]matchKind, id string) map[string]matchKind {
				if cur := acc[id]; cur < matchFamily {
					acc[id] = matchFamily
				}
				return acc
			}, matched)(A.Filter(inHUPD)(familyNorm))

			// --- Dataset row ---
			// Emit only for EPO patents that are themselves in HUPD (a direct
			// match) and cite ≥ minCollisions HUPD patents with category
			// annotations. HUPD contains only US patents, so this restricts the
			// citing side to the HUPD∩EPO intersection — i.e. US patents whose
			// examiner report collides with other HUPD patents. EPOHUPDPaths is
			// therefore always populated for emitted rows.
			if w != nil && inHUPD(epoNorm) {
				if cited := A.Filter(isHUPDCitation)(rec.Citations); len(cited) >= minCollisions {
					_, err := w.Write([]DatasetRecord{{
						EPOPatentID:  rec.PatentID,
						EPOHUPDPaths: append([]string(nil), hupdFiles[epoNorm]...),
						CitedHUPD:    A.Map(toCitedCollision)(cited),
						FamilyHUPD:   F.Pipe2(familyNorm, A.Filter(inHUPD), A.Map(toMember)),
					}})
					if err != nil {
						return err
					}
					rows++
				}
			}

			now := time.Now()
			if now.Sub(lastBar) >= hupd.BarEvery {
				lastBar = now
				rate := float64(records) / now.Sub(start).Seconds()
				mu.Lock()
				bar.Describe(fmt.Sprintf("EPO↔HUPD — records %s · matches %s · rows %s · %s",
					hupd.FmtCount(records), hupd.FmtCount(int64(len(matched))), hupd.FmtCount(rows), hupd.FmtRate(rate)))
				mu.Unlock()
			}
			if now.Sub(lastLogged) >= hupd.ProgressEvery {
				lastLogged = now
				logger.Info("analyze: EPO progress",
					"records", records, "matches", len(matched), "dataset_rows", rows)
			}
			return nil
		})
	}

	computeStats := func() overlapStats {
		vals := slices.Collect(maps.Values(matched))
		_ = matchNone
		return overlapStats{
			records:    int(records),
			direct:     len(A.Filter(P.IsStrictEqual[matchKind]()(matchDirect))(vals)),
			familyOnly: len(A.Filter(P.IsStrictEqual[matchKind]()(matchFamily))(vals)),
		}
	}

	if outPath == "" {
		return computeStats(), 0, doPass(nil)
	}

	// Dataset mode: bracket over the Parquet writer.
	writeErr := IOResF.Write[any, *os.File](IOResF.Create(outPath))(func(f *os.File) func() (any, error) {
		return func() (any, error) {
			w := parquet.NewGenericWriter[DatasetRecord](f)
			if err := doPass(w); err != nil {
				_ = w.Close()
				return nil, err
			}
			return nil, w.Close()
		}
	})
	_, err = writeErr()
	return computeStats(), rows, err
}

// Uses fp-go pipeline: Map(trim+upper) → Filter(non-empty) → StrictUniq → sort.
func uniqueUpper(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	out := F.Pipe3(
		values,
		A.Map(func(v string) string { return strings.ToUpper(strings.TrimSpace(v)) }),
		A.Filter(P.IsNonZero[string]()),
		A.StrictUniq[string],
	)
	sort.Strings(out)
	return out
}

// --- report ----------------------------------------------------------------

func newReport(epoPath, hupdDir string, hupdTotal int, ov overlapStats) Report {
	overlap := ov.direct + ov.familyOnly
	cov := 0.0
	if hupdTotal > 0 {
		cov = 100 * float64(overlap) / float64(hupdTotal)
	}
	return Report{
		EPOFile:           epoPath,
		HUPDDir:           hupdDir,
		EPORecords:        ov.records,
		HUPDTotal:         hupdTotal,
		OverlapDirect:     ov.direct,
		OverlapFamilyOnly: ov.familyOnly,
		OverlapTotal:      overlap,
		HUPDCoveragePct:   cov,
	}
}

func printReport(r Report) {
	rows := []kv{
		{"EPO file", r.EPOFile},
		{"HUPD dir", r.HUPDDir},
		{"EPO records", strconv.Itoa(r.EPORecords)},
		{"HUPD total", strconv.Itoa(r.HUPDTotal)},
		{"Overlap (direct)", strconv.Itoa(r.OverlapDirect)},
		{"Overlap (family-only)", strconv.Itoa(r.OverlapFamilyOnly)},
		{"Overlap (total)", fmt.Sprintf("%d  (%.2f%% of HUPD)", r.OverlapTotal, r.HUPDCoveragePct)},
	}
	if r.DatasetRows > 0 {
		rows = append(rows,
			kv{"Dataset rows", strconv.FormatInt(r.DatasetRows, 10)},
			kv{"Dataset file", r.DatasetFile},
		)
	}
	renderSummary("EPO ↔ HUPD analysis — summary", rows)
}

func writeJSON(path string, v any) error {
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o750); err != nil {
			return err
		}
	}
	// IOResF.Write bracket: creates the file, runs the encoder, then closes the file.
	// Replaces the manual os.Create + defer f.Close() pattern.
	_, err := IOResF.Write[any, *os.File](IOResF.Create(path))(func(f *os.File) func() (any, error) {
		return func() (any, error) {
			enc := json.NewEncoder(f)
			enc.SetIndent("", "  ")
			return nil, enc.Encode(v)
		}
	})()
	return err
}
