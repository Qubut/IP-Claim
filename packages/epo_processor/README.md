# epo-processor

> A zero-copy, resumable streaming ETL for European Patent Office (EPO) bulk
> data with built-in EPO ↔ HUPD overlap analysis.

[![Go Version](https://img.shields.io/badge/go-1.24-blue)](https://go.dev/dl/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](Makefile)
[![golangci-lint](https://img.shields.io/badge/golangci--lint-passing-brightgreen)](https://golangci-lint.run)

`epo-processor` downloads, extracts, and parses EPO patent archives from the
[EPO BDDS API](https://www.epo.org/en/searching-for-patents/data/bulk-data-sets),
streams every `<exchange-document>` node through a CPC/citation/family extractor,
and writes batched rows to Parquet. A companion `analyze` subcommand measures
how much of a local [HUPD](https://huggingface.co/datasets/HUPD/hupd) extraction
is cited in the EPO corpus.

## Features

- **Fully streaming** — tar and tar.gz archives are never written to disk; zip is
  spooled only while being walked (random-access requirement). Memory is bounded
  to one batch of patent records at a time.
- **Resumable** — a bbolt-backed checkpoint database records every fully-processed
  archive. Restarting after a crash skips already-completed work automatically.
- **Composable stages** — each pipeline stage (`Source → Opener → Extractor →
  Batch → Sink`) is an interface. Swap in `NoopSink` for dry-runs or
  `LocalFileOpener` for re-runs from disk without touching any other code.
- **Retention policies** — two independent toggles (`keep_archive`,
  `keep_extracted`) tee the raw HTTP body or selected entries to disk while the
  stream flows, at zero extra passes.
- **Overlap analysis** — `analyze` indexes a HUPD extraction (via Feather
  metadata or a JSON directory scan) and reports direct + family-graph matches
  against any EPO Parquet/CSV output.
- **Structured logging & summaries** — stdlib `log/slog` (human-readable text on
  stderr plus optional JSON log file) with fp-go logging combinators threaded in,
  and an end-of-run summary table rendered via `jedib0t/go-pretty`.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Commands](#commands)
  - [process](#process--epo-xml--parquet)
  - [process-hupd](#process-hupd--hupd-tar--disk)
  - [analyze](#analyze--epo--hupd-overlap)
- [Configuration Reference](#configuration-reference)
- [Architecture](#architecture)
- [Design Patterns](#design-patterns)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

```bash
# 1. Enter the reproducible dev environment (requires devenv / Nix)
cd packages/epo_processor
devenv shell

# 2. Build (first time only — auto-triggered by the scripts below too)
epo-build

# 3. Run the EPO streaming ETL
epo-process                          # uses config/config.yaml

# 4. Measure EPO ↔ HUPD overlap
epo-analyze \
  --dataset   data.parquet \
  --hupd-dir  /data/hupd/hupd_all-years
```

---

## Installation

### With devenv (recommended)

The project ships a `devenv.nix` that provides Go, golangci-lint, gofumpt,
goimports, golines, and all other tools in a fully reproducible Nix shell.

```bash
# Install devenv: https://devenv.sh/getting-started/
curl -L https://get.devenv.sh | bash

cd packages/epo_processor
devenv shell      # enters the Nix-managed environment
make build        # produces bin/epo-processor
```

### Without devenv

```bash
# Requires Go 1.24+
git clone https://github.com/Qubut/IP-Claim.git
cd IP-Claim/packages/epo_processor
go build -o bin/epo-processor ./cmd/epo_processor
./bin/epo-processor --help
```

### Install globally

```bash
go install github.com/Qubut/IP-Claim/packages/epo_processor/cmd/epo_processor@latest
```

---

## Commands

### `process` — EPO XML → Parquet

Downloads every archive listed in the EPO BDDS product catalogue, unwraps
nested tar/tar.gz/zip containers in-stream, parses each `<exchange-document>`
node, and batches the resulting rows into a single Parquet file.

```bash
epo-process                   # uses config/config.yaml by default
# or with explicit flags:
epo-run process --config config/config.yaml
```

| Key config field | Default | Description |
|---|---|---|
| `server.base_url` | EPO BDDS API | Root URL of the bulk-data REST service |
| `server.product_id` | `3` | EP front exchange data product |
| `server.verify_sha1` | `false` | Re-download on SHA-1 mismatch |
| `pipeline.archive_concurrency` | `4` | Parallel archives in flight |
| `pipeline.extractor_concurrency` | `4` | Parallel XML decoders per batch |
| `pipeline.output_parquet` | `./data.parquet` | Destination Parquet file |
| `pipeline.checkpoint_db` | `""` (disabled) | Path to bbolt resume database |

### `process-hupd` — HUPD .tar → disk

Streams the HUPD all-years `.tar` from HuggingFace and writes each entry to
disk. Uses the same pipeline machinery as `process` with `NoopExtractor` +
`NoopSink`; at least one retention flag must be enabled or the command refuses
to run (it would otherwise be a no-op).

```bash
epo-process-hupd              # uses config/config.yaml by default
# config.yaml: pipeline.keep_extracted: true
#              pipeline.extracted_dir:  data/hupd
```

| Key config field | Default | Description |
|---|---|---|
| `hupd.url` | HuggingFace all-years.tar | Download URL |
| `pipeline.keep_extracted` | `false` | Write each entry to `extracted_dir` |
| `pipeline.extracted_dir` | `data/xml` | Destination for extracted entries |
| `pipeline.keep_archive` | `false` | Keep the raw .tar on disk |

### `analyze` — build the EPO ↔ HUPD linked dataset

Builds a **Parquet dataset** of EPO patents that are linked to HUPD records
through direct citations or the simple family graph, annotated with citation
category codes (X, Y, A, …). Overlap statistics are reported as a JSON
summary to stdout; the dataset itself is written to `--dataset`.

The HUPD ID index is held in memory (one entry per JSON file); the EPO file
is streamed exactly once.

```bash
epo-analyze \
  --dataset   data/epo.parquet \
  --hupd-dir  /data/hupd/hupd_all-years \
  --hupd-meta /data/hupd/hupd_metadata.feather   # optional; falls back to JSON scan
```

The command prints a JSON summary to stdout:

```json
{
  "epo_file":            "data/epo.parquet",
  "hupd_dir":            "/data/hupd/hupd_all-years",
  "epo_records":         3842197,
  "hupd_total":          2923922,
  "overlap_direct":      112843,
  "overlap_family_only": 204711,
  "overlap_total":       317554,
  "hupd_coverage_pct":   10.86,
  "dataset_file":        "data/epo_hupd_dataset.parquet",
  "dataset_rows":        112843
}
```

The Parquet dataset (`--dataset`) has one row per EPO patent with ≥
`--min-collisions` cited HUPD patents; each row carries the HUPD JSON paths,
EPO citation-category codes (X, Y, A, …), and direct/family membership flags.

| Flag | Default | Description |
|---|---|---|
| `--epo` | pipeline.output_parquet | EPO input file (`.parquet`, `.csv`, `.csv.gz`) |
| `--hupd-dir` | (required) | Root of the on-disk HUPD JSON extraction |
| `--hupd-meta` | `""` | Pre-downloaded Feather file; skips the JSON scan |
| `--dataset` | `""` | **Output Parquet dataset** (one row per qualifying EPO patent) |
| `--output` | `""` | Path to write the JSON summary report (default: stdout only) |
| `--min-collisions` | `1` | Min cited-HUPD count to include an EPO patent in the dataset |
| `--hupd-workers` | `NumCPU` | Parallel JSON readers for the directory scan fallback |

**HUPD indexing strategies**

| Strategy | When to use | Speed |
|---|---|---|
| Feather (`--hupd-meta`) | Full HUPD release present | Fast (Arrow column scan) |
| JSON dir scan | Partial extraction or no Feather | Slower (parallel file I/O) |

---

## Configuration Reference

Copy `config/config.yaml` and edit as needed. Every key can also be set via
environment variable (`EPO_<SECTION>_<KEY>`, e.g.
`EPO_PIPELINE_CHECKPOINT_DB=/var/lib/epo/state.db`).

```yaml
log:
  log_level: info        # debug | info | warn | error
  log_dir:   logs        # empty → no file log

server:
  base_url:     "https://publication-bdds.apps.epo.org/bdds/bdds-bff-service/prod/api/public"
  product_id:   3        # EP full-text exchange data
  max_retries:  5        # per-archive HTTP retry limit (0–10)
  timeout:      30s
  verify_sha1:  true     # re-download on checksum mismatch

hupd:
  url:      "https://huggingface.co/datasets/HUPD/hupd/resolve/main/data/all-years.tar"
  filename: data/hupd_all-years.tar

analyze:
  hupd_meta_url: "https://huggingface.co/datasets/HUPD/hupd/resolve/main/hupd_metadata_2022-02-22.feather"

pipeline:
  archive_concurrency:   4           # parallel archive downloads
  extractor_concurrency: 4           # parallel XML decoders
  batch_size:            1000        # rows per Parquet write
  batch_timeout:         2s          # partial-batch idle flush
  output_parquet:        ./data.parquet
  row_group_size:        50000       # Parquet row group (bounds RAM)
  spool_dir:             ""          # zip spool — defaults to OS temp
  use_local_dir:         ""          # replay from disk instead of HTTP

  keep_archive:    false             # tee raw HTTP body → archive_dir
  archive_dir:     data/archives
  keep_extracted:  false             # tee selected entries → extracted_dir
  extracted_dir:   data/xml

  checkpoint_db:      data/.epo-state.db   # empty → disable resume
  reset_checkpoint:   false                # wipe state + output on startup
```

---

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐    ┌─────────┐    ┌────────────┐
│ArchiveSource │───▶│ArchiveOpener │───▶│ RecordExtractor  │───▶│  Batch  │───▶│ RecordSink │
│  (catalogue) │    │ (HTTP/local) │    │  (XML parser)    │    │ (rill)  │    │  (Parquet) │
└──────────────┘    └──────────────┘    └──────────────────┘    └─────────┘    └────────────┘
       │                   │                     │
  EPO product         download +           one PatentRecord
  catalogue JSON      unwrap nested        per <exchange-document>
  → ArchiveJob        tar/gz/zip           node (CPC, citations,
    per .tar          → XMLEntry           family members)
```

**Stage implementations per subcommand:**

| Stage | Interface | `process` | `process-hupd` |
|---|---|---|---|
| Source | `ArchiveSource` | `EPOProductSource` | `StaticListSource` |
| Opener | `ArchiveOpener` | `HTTPOpener` | `HTTPOpener` |
| Extractor | `RecordExtractor` | `XMLStreamExtractor` | `NoopExtractor` |
| Sink | `RecordSink` | `ParquetSink` | `NoopSink` |

All stage-to-stage communication uses typed channels from
[`destel/rill`](https://github.com/destel/rill). Cancellation and backpressure
propagate through every stage via `context.Context`.

---

## Development

### Enter the dev environment

```bash
cd packages/epo_processor
devenv shell          # activates the Nix-managed shell with all tools
```

Once inside the shell, the following convenience scripts are available
directly on `$PATH` (defined in `devenv.nix`):

| Script | Description |
|---|---|
| `epo-build` | `make build` — compiles `bin/epo-processor` |
| `epo-run [args]` | build if needed, then `epo-processor [args]` |
| `epo-process [args]` | `epo-processor process --config config/config.yaml [args]` |
| `epo-process-hupd [args]` | `epo-processor process-hupd --config config/config.yaml [args]` |
| `epo-analyze [args]` | `epo-processor analyze --config config/config.yaml [args]` |
| `epo-test` | `go test -race ./...` |

### Makefile targets

| Target | Description |
|---|---|
| `make build` | Build `bin/epo-processor` |
| `make test` | Run tests with `-race` |
| `make test-cover` | Open coverage HTML report in browser |
| `make lint` | `golangci-lint run ./...` |
| `make fmt` | `gofumpt` + `goimports` + `golines` |
| `make tidy` | `go mod tidy && go mod verify` |
| `make build-all` | Cross-compile: linux/darwin/windows × amd64/arm64 |
| `make clean` | Remove `bin/` and `coverage.out` |
| `make dev` | Live reload via `air` |
| `make help` | List all targets |

The default target (`make`) runs `tidy → lint → test → build`.

### Browse the API docs locally

```bash
go install golang.org/x/pkgsite/cmd/pkgsite@latest
pkgsite -http :6060 &
# open http://localhost:6060/github.com/Qubut/IP-Claim/packages/epo_processor
```

---

## Contributing

1. Fork the repository and create a feature branch.
2. `devenv shell` to get all tools.
3. `make all` (fmt → tidy → lint → test → build) must pass with 0 lint issues.
4. Open a pull request with a concise description of the change and the
   motivation behind it.

Commit style: `<scope>: <imperative verb> <what>` (e.g. `pipeline: add tee for kept entries`).

---

## License

MIT — see [LICENSE](LICENSE).

---

## Related projects and data sources

| Resource | URL |
|---|---|
| EPO BDDS API | https://www.epo.org/en/searching-for-patents/data/bulk-data-sets |
| HUPD dataset | https://huggingface.co/datasets/HUPD/hupd |
| destel/rill | https://github.com/destel/rill |
| IBM/fp-go | https://github.com/IBM/fp-go |
| parquet-go | https://github.com/parquet-go/parquet-go |
| bbolt | https://github.com/etcd-io/bbolt |
| jedib0t/go-pretty | https://github.com/jedib0t/go-pretty |
