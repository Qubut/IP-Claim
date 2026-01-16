# ==============================================================================
# Makefile for epo-processor CLI
# ==============================================================================

APP_NAME     := epo-processor
BINARY_NAME  := $(APP_NAME)
MAIN_PACKAGE := ./cmd/$(APP_NAME)
VERSION      := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
COMMIT       := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_TIME   := $(shell date -u '+%Y-%m-%dT%H:%M:%SZ')
GO_VERSION   := $(shell go version | cut -d' ' -f3)

# Build flags for embedding version information
LDFLAGS := -X 'github.com/Qubut/IP-Claim/packages/epo_processor/cmd.Version=$(VERSION)' \
          -X 'github.com/Qubut/IP-Claim/packages/epo_processor/cmd.Commit=$(COMMIT)' \
          -X 'github.com/Qubut/IP-Claim/packages/epo_processor/cmd.BuildTime=$(BUILD_TIME)' \
          -X 'github.com/Qubut/IP-Claim/packages/epo_processor/cmd.GoVersion=$(GO_VERSION)'

# Common flags
GOFLAGS      := -trimpath
BUILD_FLAGS  := -ldflags "$(LDFLAGS)" $(GOFLAGS)

# Colors for output
CYAN         := \033[0;36m
GREEN        := \033[0;32m
RED          := \033[0;31m
NC           := \033[0m # No Color

# ==============================================================================
# Targets
# ==============================================================================

.PHONY: all
all: tidy lint test build

.PHONY: build
build:
	@echo "$(CYAN)→ Building $(APP_NAME) $(VERSION)$(NC)"
	@CGO_ENABLED=0 go build $(BUILD_FLAGS) -o bin/$(BINARY_NAME) $(MAIN_PACKAGE)

.PHONY: build-all
build-all: ## Cross-compile for common platforms
	@echo "$(CYAN)→ Cross-compiling for common platforms$(NC)"
	@mkdir -p bin
	@GOOS=linux   GOARCH=amd64   CGO_ENABLED=0 go build $(BUILD_FLAGS) -o bin/$(BINARY_NAME)-linux-amd64   $(MAIN_PACKAGE)
	@GOOS=linux   GOARCH=arm64   CGO_ENABLED=0 go build $(BUILD_FLAGS) -o bin/$(BINARY_NAME)-linux-arm64   $(MAIN_PACKAGE)
	@GOOS=darwin  GOARCH=amd64   CGO_ENABLED=0 go build $(BUILD_FLAGS) -o bin/$(BINARY_NAME)-darwin-amd64  $(MAIN_PACKAGE)
	@GOOS=darwin  GOARCH=arm64   CGO_ENABLED=0 go build $(BUILD_FLAGS) -o bin/$(BINARY_NAME)-darwin-arm64  $(MAIN_PACKAGE)
	@GOOS=windows GOARCH=amd64   CGO_ENABLED=0 go build $(BUILD_FLAGS) -o bin/$(BINARY_NAME)-windows-amd64.exe $(MAIN_PACKAGE)

.PHONY: run
run: build ## Build and run the application
	@echo "$(GREEN)→ Running $(APP_NAME)$(NC)"
	@./bin/$(BINARY_NAME) --help

.PHONY: test
test: ## Run tests with race detector
	@echo "$(CYAN)→ Running tests with race detector$(NC)"
	@go test -race -count=1 -coverprofile=coverage.out ./...

.PHONY: test-cover
test-cover: test ## Run tests and show coverage in browser
	@echo "$(CYAN)→ Opening coverage report$(NC)"
	@go tool cover -html=coverage.out

.PHONY: tidy
tidy: ## Tidy go modules
	@echo "$(CYAN)→ Tidying modules$(NC)"
	@go mod tidy
	@go mod verify
.PHONY: fmt
fmt: ## Format code with gofumpt + goimports + golines (modern strict & short lines)
	@echo "$(CYAN)→ Formatting code (gofumpt + goimports + golines)...$(NC)"
	# 1. gofumpt (stricter gofmt)
	@gofumpt -w -extra ./cmd ./internal ./packages
	# 2. goimports (organize imports, local package grouping)
	@goimports -w -local github.com/Qubut/IP-Claim/packages/epo_processor ./cmd ./internal ./packages
	# 3. golines (shorten long lines – max 100 or 120 chars)
	@golines -w --shorten-comments --max-len=100 --base-formatter=goimports ./cmd ./internal ./packages
	@echo "$(GREEN)Formatting complete$(NC)"
.PHONY: lint
lint: ## Run golangci-lint (install if needed)
	@echo "$(CYAN)→ Running golangci-lint$(NC)"
	@golangci-lint run --timeout=5m ./...

.PHONY: clean
clean: ## Clean build artifacts
	@echo "$(RED)→ Cleaning...$(NC)"
	@rm -rf bin/ coverage.out

.PHONY: dev
dev: ## Run with air for live reload (requires air)
	@if command -v air >/dev/null 2>&1; then \
		echo "$(GREEN)→ Starting live reload with air$(NC)"; \
		air; \
	else \
		echo "$(RED)Error:$(NC) 'air' is not installed."; \
		echo "Install it with: go install github.com/air-verse/air@latest"; \
		exit 1; \
	fi

.PHONY: help
help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ==============================================================================
# Release helpers (optional – for advanced usage)
# ==============================================================================

.PHONY: release
release: ## Create GitHub release with goreleaser (requires configuration)
	@echo "$(CYAN)→ Suggested: use goreleaser for professional releases$(NC)"
	@echo "Install: go install github.com/goreleaser/goreleaser@latest"
	@echo "Then run: goreleaser release --clean"

.DEFAULT_GOAL := help
