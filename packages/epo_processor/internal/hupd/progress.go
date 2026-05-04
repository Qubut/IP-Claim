package hupd

import "time"

// ProgressEvery throttles progress log lines.
const ProgressEvery = 5 * time.Second

// ProgressReporter is the minimal interface for displaying scan progress.
// cmd.progressBar and cmd.noopProgressBar both satisfy it.
type ProgressReporter interface {
	Describe(string)
	Set64(int64) error
}
