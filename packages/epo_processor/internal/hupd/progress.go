package hupd

import (
	"strconv"
	"strings"
	"time"
)

// ProgressEvery throttles progress log lines.
const ProgressEvery = 5 * time.Second

// BarEvery throttles live spinner-message updates. Far tighter than
// ProgressEvery so the displayed counts move smoothly while log lines stay
// sparse.
const BarEvery = 250 * time.Millisecond

// ProgressReporter is the minimal interface for displaying scan progress.
// cmd.progressBar and cmd.noopProgressBar both satisfy it. All insight is
// carried in the Describe message.
type ProgressReporter interface {
	Describe(string)
}

// FmtCount renders n with thousands separators, e.g. 1234567 -> "1,234,567".
func FmtCount(n int64) string {
	s := strconv.FormatInt(n, 10)
	neg := strings.HasPrefix(s, "-")
	if neg {
		s = s[1:]
	}
	var b strings.Builder
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(c)
	}
	if neg {
		return "-" + b.String()
	}
	return b.String()
}

// FmtRate renders a per-second rate compactly, e.g. 12345 -> "12.3K/s".
func FmtRate(perSec float64) string {
	switch {
	case perSec >= 1_000_000:
		return strconv.FormatFloat(perSec/1_000_000, 'f', 1, 64) + "M/s"
	case perSec >= 1_000:
		return strconv.FormatFloat(perSec/1_000, 'f', 1, 64) + "K/s"
	default:
		return strconv.FormatFloat(perSec, 'f', 0, 64) + "/s"
	}
}
