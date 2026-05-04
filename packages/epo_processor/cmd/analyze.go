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
	Use:   "analyze",
	Short: "Build a linked EPO↔HUPD Parquet dataset and report citation overlap",
	RunE: func(cmd *cobra.Command, _ []string) error {
		ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer cancel()

		epoPath, _ := cmd.Flags().GetString("epo")
		hupdDir, _ := cmd.Flags().GetString("hupd-dir")
		metaPath, _ := cmd.Flags().GetString("hupd-meta")
		outputJSON, _ := cmd.Flags().GetString("output")
		datasetPath, _ := cmd.Flags().GetString("dataset")
		minCollisions, _ := cmd.Flags().GetInt("min-collisions")
		workers, _ := cmd.Flags().GetInt("hupd-workers")
		if workers <= 0 {
			workers = runtime.NumCPU()
		}

		if strings.TrimSpace(epoPath) == "" {
			epoPath = cfg.Pipeline.OutputParquet
		}
		if strings.TrimSpace(epoPath) == "" {
			return fmt.Errorf("--epo is required (or set pipeline.output_parquet)")
		}
		if strings.TrimSpace(hupdDir) == "" {
			return fmt.Errorf("--hupd-dir is required")
		}
		if minCollisions < 1 {
			minCollisions = 1
		}

		hupdStart := time.Now()
		bar := newProgressBar("[cyan]analyze[reset]")
		var barMu sync.Mutex
		logger.Infow("analyze: scanning HUPD", "dir", hupdDir, "meta", metaPath, "workers", workers)
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
			_ = bar.Finish()
			return fmt.Errorf("scan HUPD: %w", err)
		}
		logger.Infow("analyze: HUPD scanned",
			"ids", len(hupdFiles),
			"elapsed", time.Since(hupdStart).String(),
		)

		epoStart := time.Now()
		logger.Infow("analyze: streaming EPO (single pass)", "path", epoPath, "dataset", datasetPath)
		ov, datasetRows, err := streamEPOOnce(ctx, epoPath, hupdFiles, datasetPath, minCollisions, bar, &barMu)
		if err != nil {
			_ = bar.Finish()
			return fmt.Errorf("stream EPO: %w", err)
		}
		logger.Infow("analyze: EPO pass done",
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
			logger.Infow("analyze: report written", "path", outputJSON)
		}

		_ = bar.Finish()
		printReport(report)
		return nil
	},
}

func init() {
	analyzeCmd.Flags().String("epo", "", "EPO records file (.parquet, .csv, .csv.gz). Defaults to pipeline.output_parquet.")
	analyzeCmd.Flags().String("hupd-dir", "", "Root directory of extracted HUPD JSON files (recursive).")
	analyzeCmd.Flags().String("hupd-meta", "", "Path to a pre-downloaded HUPD Feather metadata file. If absent, falls back to full JSON dir scan.")
	analyzeCmd.Flags().String("output", "", "Optional path to write the full report as JSON.")
	analyzeCmd.Flags().String("dataset", "", "Path to write a Parquet dataset (one DatasetRecord per EPO patent with ≥ min-collisions cited-HUPD patents with category annotations).")
	analyzeCmd.Flags().Int("min-collisions", 1, "Min cited-HUPD-with-categories count per EPO patent to emit a TrainingRecord (default 1).")
	analyzeCmd.Flags().Int("hupd-workers", 0, "Concurrent HUPD JSON readers (0 = NumCPU).")
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
	var lastLogged time.Time

	mu.Lock()
	bar.Describe("[cyan]EPO↔HUPD[reset]")
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
			// Emit for any EPO patent that cites ≥ minCollisions HUPD patents
			// with category annotations. EPOHUPDPaths is non-nil only when the
			// EPO patent is itself in HUPD (direct match).
			if w != nil {
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

			if now := time.Now(); now.Sub(lastLogged) >= hupd.ProgressEvery {
				lastLogged = now
				mu.Lock()
				_ = bar.Set64(records)
				mu.Unlock()
				logger.Infow("analyze: EPO progress",
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

// Uses fp-go pipeline: Map(trim+upper) → Filter(non-empty+unseen) → sort.
func uniqueUpper(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(values))
	isUnseen := func(v string) bool {
		if _, ok := seen[v]; ok {
			return false
		}
		seen[v] = struct{}{}
		return true
	}
	filtered := F.Pipe2(
		values,
		A.Map(func(v string) string { return strings.ToUpper(strings.TrimSpace(v)) }),
		A.Filter(F.Pipe1(P.IsNonZero[string](), P.And(isUnseen))),
	)
	sort.Strings(filtered)
	return filtered
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
	fmt.Println("===== EPO ↔ HUPD analysis =====")
	fmt.Printf("  EPO file           : %s\n", r.EPOFile)
	fmt.Printf("  HUPD dir           : %s\n", r.HUPDDir)
	fmt.Printf("  EPO records        : %d\n", r.EPORecords)
	fmt.Printf("  HUPD total         : %d\n", r.HUPDTotal)
	fmt.Println("--- overlap ---")
	fmt.Printf("  direct match       : %d\n", r.OverlapDirect)
	fmt.Printf("  family-only match  : %d\n", r.OverlapFamilyOnly)
	fmt.Printf("  total in EPO       : %d  (%.2f%% of HUPD)\n", r.OverlapTotal, r.HUPDCoveragePct)
	if r.DatasetRows > 0 {
		fmt.Println("--- dataset ---")
		fmt.Printf("  rows written       : %d\n", r.DatasetRows)
		fmt.Printf("  output file        : %s\n", r.DatasetFile)
	}
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
