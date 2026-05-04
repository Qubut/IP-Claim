// Package telemetry bootstraps OpenTelemetry (logs, traces, metrics) for
// the processor and exposes a single shutdown hook for graceful teardown.
//
// Telemetry is opt-in via the YAML field telemetry.enabled. When
// disabled, cmd uses no-op tracer and meter providers and a plain zap
// logger, so nothing in this package executes.
//
// When enabled, log / trace / metric exporters are wired over either
// gRPC or HTTP, selected by the protocol field. Endpoint, headers, and
// resource attributes come from the [Config].
package telemetry
