# EPO Processor CLI

A command-line tool built in Go for downloading, extracting, and parsing European Patent Office (EPO) patent data. It supports concurrent processing, configurable telemetry (OTEL), and functional programming patterns via fp-go, with built-in metrics, tracing, and CSV output.

![Go Version](https://img.shields.io/badge/go-1.22-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

- Go 1.25+ (download from https://go.dev/dl/)
- Make (standard on Unix, install via Chocolatey on Windows)
- Optional: goreleaser for releases (go install github.com/goreleaser/goreleaser@latest)
- For full functionality: Access to EPO API (configure in config.yaml)

Install dependencies:

```bash
go mod download
```

### Installation

1. Clone the repository:

```bash

git clone https://github.com/Qubut/IP-Claim.git

cd packages/epo_processor

```

2. Build the binary:

```bash

make build

```

3. Or install globally:

```bash

go install ./cmd/epo-processor

```

4. Configure: Copy `config/example.yaml` to `config/config.yaml` and edit (e.g., server.base_url, telemetry.endpoint).

### Running the CLI

Basic usage:

```bash

epo-processor --config config/config.yaml

# Runs all enabled steps: download, extract, parse

```

Subcommands:

```bash

epo-processor download   # Only download

epo-processor extract    # Only extract

epo-processor parse      # Only parse

epo-processor version    # Show version

epo-processor config print  # Print loaded config

```

For help:

```bash

epo-processor --help

epo-processor [command] --help

```

## Makefile

The Makefile provides a complete build pipeline. Run commands from the project root.

Run build with tests (default target):

```bash

make all   # fmt + tidy + lint + test + build

```

Format code:

```bash

make fmt

```

Build the application:

```bash

make build

```

Run the application:

```bash

make run

```

Live reload for development:

```bash

make dev  

```

Run the test suite:

```bash

make test

```

Clean up binary and artifacts:

```bash

make clean

```

Cross-compile for multiple platforms:

```bash

make build-all

```

Show all commands:

```bash

make help

```

## Deployment

For production:

- Build the binary: `make build-all` for platform-specific executables.

- Distribute via goreleaser: Configure `.goreleaser.yml`, then `make release`.

- Docker: Add a Dockerfile:

```dockerfile

FROM golang:1.22 AS builder

WORKDIR /app

COPY . .

RUN make build

FROM scratch

COPY --from=builder /app/bin/epo-processor /

ENTRYPOINT ["/epo-processor"]

```

Build: `docker build -t epo-processor .`

Run: `docker run --rm -v $(pwd)/config:/config -v $(pwd)/data:/data epo-processor --config /config/config.yaml`

- Kubernetes: Use ConfigMap for config.yaml, mount volumes for data.

## Troubleshooting

- Config errors: Check unmarshal issues (use `epo-processor config print`).

- No citations: Ensure XML has <patcit>; debug with logs.

- Permissions: Run as non-root; check file owners.

## License

MIT License – see LICENSE file.
