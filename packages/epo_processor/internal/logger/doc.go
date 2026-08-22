// Package logger provides a thin factory around the standard library
// [log/slog] for building the application logger.
//
// [New] returns a logger that writes human-readable text to stderr and,
// when a log directory is given, structured JSON to a timestamped main log
// plus a WARN+ errors log in that directory. A small fan-out handler
// multiplexes records to every sink without any third-party dependency,
// keeping stdout free for data output and the progress bar.
//
// Alongside the logger it returns [Controls], which expose:
//
//   - Console: a steerable console handler (writer + level). During an
//     interactive run a command calls Console.SetTTYMode to route console
//     output through the live progress display and raise the threshold to
//     WARN (INFO still flows to the file), then Console.Restore afterwards.
//   - Errors: a [Collector] slog.Handler that records every WARN+ event in
//     memory so the command can show counts and affected archives in its
//     end-of-run summary.
//   - ErrorsPath: the on-disk WARN+ errors log (empty when no log dir).
//
// Levels accepted (case-insensitive): debug, info, warn, error.
package logger
