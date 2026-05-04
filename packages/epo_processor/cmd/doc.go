// Package cmd holds the cobra root command and all subcommands for the
// epo-processor binary.
//
// # Overview
//
// Each subcommand in this package composes the same five-stage streaming
// pipeline from [github.com/Qubut/IP-Claim/packages/epo_processor/internal/pipeline]
// using Functional Options. The shared concerns — config loading, logging,
// OpenTelemetry bootstrap, signal handling — are wired once in root.go and
// made available to every subcommand via cobra's PersistentPreRunE hook.
//
// # Subcommands
//
// process — EPO XML → Parquet
//
// Downloads every archive listed in the EPO BDDS product catalogue, unwraps
// nested tar/gz/zip containers in-stream, parses each <exchange-document>
// node with [github.com/Qubut/IP-Claim/packages/epo_processor/internal/parse.ExtractPatentRecord],
// and writes batched rows to a Parquet file. Checkpointing via bbolt lets
// interrupted runs resume without reprocessing completed archives.
//
// process-hupd — HUPD .tar → disk
//
// Streams the HUPD all-years .tar from HuggingFace and writes each entry to
// disk using the same pipeline machinery with [pipeline.NoopExtractor] and
// [pipeline.NoopSink]. Requires at least one retention flag
// (keep_archive or keep_extracted) or returns an error.
//
// analyze — build the EPO ↔ HUPD linked Parquet dataset
//
// Builds a Parquet dataset where each row is an EPO patent linked to one or
// more HUPD records via direct citation or the simple family graph. Each row
// carries the HUPD JSON paths and EPO citation-category codes (X, Y, A, …),
// making the dataset directly usable for downstream ML tasks. Overlap
// statistics are printed to stdout as a JSON summary.
//
// The HUPD ID index is held in memory (Feather Arrow scan or parallel JSON
// walk); the EPO file is streamed exactly once.
//
// # Pipeline architecture
//
// All subcommands share the same interface composition:
//
//	ArchiveSource → ArchiveOpener → RecordExtractor → Batch → RecordSink
//
// Stages communicate over typed rill channels with backpressure. Cancellation
// propagates via context.Context. Each stage can be replaced by a Noop
// implementation without touching any other stage.
//
// # Shared wiring (root.go)
//
//  1. [internal/config.LoadConfig] merges YAML + env-vars.
//  2. A zap development logger is attached to stderr (console).
//  3. A rotating JSON file logger is attached when log.log_dir is set.
//  4. OpenTelemetry is bootstrapped when telemetry.enabled is true.
//  5. os.Signal listeners for SIGINT / SIGTERM cancel the pipeline context.
//
// # Design patterns
//
// Functional Options ([pipeline.WithSource], [pipeline.WithSink], …) are
// used to construct every pipeline. Null-Object implementations
// ([pipeline.NoopSink], [pipeline.NoopCheckpointer], etc.) eliminate
// nil-guard branches throughout. The Decorator pattern
// ([pipeline.CheckpointSource]) wraps sources with resume logic without
// modifying the underlying type.
package cmd
