// Package parse maps EPO XML <exchange-document> nodes to flat
// [PatentRecord] values for downstream Parquet storage.
//
// All exported functions are pure and goroutine-safe. The streaming
// pipeline calls [ExtractPatentRecord] per element from a pool of
// extractor goroutines without any synchronisation.
//
// Per-section parsers (classifications, citations, family members)
// degrade to an empty slice on malformed sub-trees rather than aborting
// the whole record. Hard failures (missing required root attributes)
// still bubble up as a returned error.
package parse
