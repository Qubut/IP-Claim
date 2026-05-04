// Package hupd provides utilities for building the HUPD patent ID index
// used by the analyze subcommand.
//
// HUPD (Harvard USPTO Patent Dataset) distributes patent records as
// individual JSON files, one per application, organised under
//
//	<year>/<year>/<application_number>.json
//
// The package supports two indexing strategies depending on what data
// is available locally:
//
//   - Feather metadata scan ([ScanMeta]): reads the pre-built Feather
//     file published alongside the dataset on HuggingFace. Column reads
//     are performed with the Apache Arrow IPC reader; only four columns
//     are consumed (application_number, patent_number, publication_number,
//     filing_date), so memory usage is proportional to the index size, not
//     the full record payloads.
//
//   - JSON directory scan ([ScanIDs]): walks the on-disk directory tree
//     and parses the patent_number / publication_number header fields
//     from each JSON file in parallel. Use this when the Feather file is
//     unavailable or when the on-disk dataset is a subset of the full
//     release.
//
// Both strategies produce the same output: a map from normalised US
// patent identifier to one or more JSON file paths. Normalisation strips
// the country prefix ("US"), any trailing kind code ("B2", "A1"), and the
// date suffix HUPD appends to publication_number values ("-YYYYMMDD").
// The resulting bare numeric string matches the EPO record's patent_id
// after the same transformation is applied by [NormalizeUSID].
//
// # Typical call sequence
//
//	ids, err := hupd.ScanMeta(ctx, metaPath, metaURL, hupdDir, bar, &mu, log)
//	// or, without Feather:
//	ids, err := hupd.ScanIDs(ctx, hupdDir, workers, bar, &mu, log)
//
// # Thread safety
//
// [ScanIDs] and [ScanMeta] are safe to call from any goroutine. The bar
// and mu parameters are used only for progress reporting and must outlive
// the call. The returned map is created fresh on each call and is not
// shared internally.
package hupd
