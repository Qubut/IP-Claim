// Package logger provides a thin factory around [go.uber.org/zap] for
// building file-only structured loggers.
//
// It is intentionally minimal: console / TTY logging is built separately
// in cmd (see buildConsoleLogger) so that the file logger can be reused
// from headless contexts (tests, batch jobs, CI) without dragging in
// terminal-detection dependencies.
//
// # Behaviour
//
//   - Empty logPath  → no-op logger ([zap.NewNop]).
//   - Unknown level  → error.
//   - Otherwise      → JSON encoder, ISO8601 timestamps, level-filtered.
//
// Levels accepted (case-insensitive): debug, info, warn, error.
package logger
