// Package config loads and validates the YAML / environment-variable
// configuration for the streaming EPO/HUPD processor.
//
// # Subcommands
//
// Two streaming subcommands share one [Config] tree:
//
//   - process       — EPO XML  → Parquet
//   - process-hupd  — HUPD .tar → disk
//
// The legacy download / extract / parse chain has been removed; only the
// fields used by the streaming pipeline are recognised.
//
// # Loading Order
//
//  1. Built-in defaults ([Config] zero values + viper.SetDefault).
//  2. YAML file (path supplied to [LoadConfig], typically config/config.yaml).
//  3. Environment variables (uppercase, dot → underscore).
//
// # Validation
//
// Every [LoadConfig] call runs the struct through go-playground/validator
// and fails fast on missing required fields. Invariants beyond struct
// tags (e.g. spool dir writability) are checked in cmd at start-up.
//
// # Environment Variables
//
// All keys are bindable. Examples:
//
//	EPO_PROCESSOR_PIPELINE_CHECKPOINT_DB=/var/lib/epo/state.db
//	EPO_PROCESSOR_TELEMETRY_ENABLED=true
package config
