package cmd

// --- EPO overlap models ----------------------------------------------------

// matchKind classifies the best EPO match for one HUPD ID.
type matchKind uint8

const (
	matchNone   matchKind = 0
	matchFamily matchKind = 1
	matchDirect matchKind = 2 // wins over Family
)

// overlapStats holds the result of one EPO overlap pass.
type overlapStats struct {
	records    int
	direct     int
	familyOnly int
}

// --- Dataset models --------------------------------------------------------

// CollisionCitation is a HUPD patent cited by an EPO patent with at least
// one EPO examiner category (X, Y, A, …). CitedID is US-normalised.
type CollisionCitation struct {
	// CitedID is the bare US numeric identifier (e.g. "9114971"), obtained
	// by stripping the "US" prefix and kind code from the EPO citation.
	CitedID string `json:"cited_id" parquet:"cited_id"`
	// Categories are the EPO relevance codes for this citation
	// (X, Y, A, E, O, T, …) as extracted from <category> nodes.
	Categories []string `json:"categories" parquet:"categories,list"`
	// Paths are the on-disk paths of the matching HUPD JSON files.
	Paths []string `json:"paths" parquet:"paths,list"`
}

// HUPDMember is a HUPD patent found in the EPO patent's family graph.
type HUPDMember struct {
	// ID is the US-normalised family-member identifier.
	ID string `json:"id" parquet:"id"`
	// Paths are the on-disk paths of the matching HUPD JSON files.
	Paths []string `json:"paths" parquet:"paths,list"`
}

// DatasetRecord is one row in the EPO↔HUPD dataset (Parquet or JSONL).
// EPOHUPDPaths is empty when the EPO patent is not itself in HUPD.
// CitedHUPD is guaranteed non-empty (len ≥ minCollisions).
type DatasetRecord struct {
	// EPOPatentID is the full EPO identifier (country+number+kind, e.g. "EP1234567A1").
	EPOPatentID string `json:"epo_patent_id" parquet:"epo_patent_id"`
	// EPOHUPDPaths are the HUPD JSON file paths for this EPO patent itself,
	// populated only when it appears in the HUPD index.
	EPOHUPDPaths []string `json:"epo_hupd_paths" parquet:"epo_hupd_paths,list"`
	// CitedHUPD lists every HUPD patent cited in the EPO examination report
	// with at least one category annotation. Guaranteed non-empty.
	CitedHUPD []CollisionCitation `json:"cited_hupd" parquet:"cited_hupd,list"`
	// FamilyHUPD lists HUPD patents that belong to the same simple family
	// as the EPO patent but were not directly cited.
	FamilyHUPD []HUPDMember `json:"family_hupd" parquet:"family_hupd,list"`
}

// Report is the analyze command's structured output, printed as JSON.
type Report struct {
	EPOFile string `json:"epo_file"` // path of the EPO input file
	HUPDDir string `json:"hupd_dir"` // root of the on-disk HUPD extraction
	// EPORecords is the total number of EPO patent records scanned.
	EPORecords int `json:"epo_records"`
	// HUPDTotal is the number of unique normalised HUPD IDs in the index.
	HUPDTotal int `json:"hupd_total"`
	// OverlapDirect is the count of HUPD IDs that matched an EPO patent_id directly.
	OverlapDirect int `json:"overlap_direct"`
	// OverlapFamilyOnly is the count of HUPD IDs reached only via the EPO family graph.
	OverlapFamilyOnly int `json:"overlap_family_only"`
	// OverlapTotal is OverlapDirect + OverlapFamilyOnly.
	OverlapTotal int `json:"overlap_total"`
	// HUPDCoveragePct is OverlapTotal / HUPDTotal × 100.
	HUPDCoveragePct float64 `json:"hupd_coverage_pct"`
	// Dataset fields are populated only when --out is set.
	DatasetRows int64  `json:"dataset_rows,omitempty"`
	DatasetFile string `json:"dataset_file,omitempty"`
}
