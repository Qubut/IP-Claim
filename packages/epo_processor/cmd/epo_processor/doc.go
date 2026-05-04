// Command epo-processor is a zero-copy, resumable streaming ETL for
// European Patent Office (EPO) bulk data with built-in EPO ↔ HUPD
// overlap analysis.
//
// This package contains only main(), which delegates to
// [github.com/Qubut/IP-Claim/packages/epo_processor/cmd].Execute.
// All flag handling, config loading, logging, and pipeline wiring live
// in the cmd package.
//
// # Quick start (devenv / Nix — recommended)
//
// devenv.nix exposes first-class convenience scripts on $PATH:
//
//	cd packages/epo_processor
//	devenv shell            # enter the Nix-managed dev environment
//
//	epo-build               # compile bin/epo-processor (auto-invoked below)
//	epo-process             # stream EPO XML → Parquet  (uses config/config.yaml)
//	epo-process-hupd        # download HUPD .tar → disk
//	epo-analyze [flags]     # build EPO↔HUPD linked dataset
//	epo-test                # go test -race ./...
//
// # Quick start (plain Go)
//
//	go build -o bin/epo-processor ./cmd/epo_processor
//	./bin/epo-processor --help
//
// # Subcommands
//
// process — stream EPO XML archives to Parquet
//
//	epo-process                          # devenv
//	./bin/epo-processor process \        # plain Go
//	    --config config/config.yaml
//
// Downloads every archive from the EPO BDDS product catalogue (product 3
// by default), unwraps nested tar/gz/zip containers entirely in-stream,
// parses each <exchange-document> node, and writes batched rows to a
// Parquet file. A bbolt checkpoint database enables crash-resume.
//
// process-hupd — download HUPD all-years.tar to disk
//
//	epo-process-hupd                     # devenv
//	./bin/epo-processor process-hupd \   # plain Go
//	    --config config/config.yaml
//	# config.yaml must set: pipeline.keep_extracted: true
//
// Streams the HUPD HuggingFace tarball and materialises each entry under
// pipeline.extracted_dir. Uses the same pipeline with NoopExtractor +
// NoopSink. Fails if no retention flag is set.
//
// analyze — build the EPO ↔ HUPD linked Parquet dataset
//
//	epo-analyze \                        # devenv
//	    --epo        data/epo.parquet \
//	    --hupd-dir   /data/hupd/hupd_all-years \
//	    --dataset    data/epo_hupd_dataset.parquet
//
// Builds a Parquet dataset of EPO patents linked to HUPD records via direct
// citation or the simple family graph, annotated with EPO citation-category
// codes (X, Y, A, …). Each row is a DatasetRecord ready for downstream ML.
// Overlap statistics are also printed to stdout as a JSON summary.
//
// # Configuration
//
// All subcommands share a single YAML config file (--config flag).
// Every key is also settable via environment variable:
//
//	EPO_PROCESSOR_PIPELINE_CHECKPOINT_DB=/var/lib/epo/state.db
//	EPO_PROCESSOR_TELEMETRY_ENABLED=true
//
// See config/config.yaml for the annotated full schema.
//
// # Installation
//
//	go install github.com/Qubut/IP-Claim/packages/epo_processor/cmd/epo_processor@latest
//
// Or with devenv (reproduces the exact toolchain used in CI):
//
//	cd packages/epo_processor && devenv shell && epo-build
package main
