// Package models defines the EPO Open Patent Services (OPS) catalogue
// JSON schema: [Product] → [Delivery] → [Item] → archive download URL.
//
// These types mirror the JSON returned by the EPO bulk-data product
// catalogue and are consumed by the pipeline source to enumerate
// [github.com/Qubut/IP-Claim/packages/epo_processor/internal/pipeline.ArchiveJob]
// units of work.
//
// # Stability
//
// The shape is dictated by EPO and changes only when EPO ships a new
// catalogue version. Adding fields is safe (json decoder ignores
// unknown keys); renaming requires a coordinated update.
package models
