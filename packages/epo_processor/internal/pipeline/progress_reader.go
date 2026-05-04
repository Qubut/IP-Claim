package pipeline

import (
	"io"
	"time"
)

// progressReader wraps r and invokes onTick with the running byte count
// at most once per interval. The final count is always emitted on EOF.
//
// onTick must not block; it is called from Read on the producer goroutine.
type progressReader struct {
	r        io.Reader
	total    int64
	read     int64
	last     time.Time
	interval time.Duration
	onTick   func(read, total int64)
}

func newProgressReader(r io.Reader, total int64, interval time.Duration, onTick func(read, total int64)) *progressReader {
	if interval <= 0 {
		interval = 100 * time.Millisecond
	}
	return &progressReader{r: r, total: total, interval: interval, onTick: onTick}
}

func (p *progressReader) Read(buf []byte) (int, error) {
	n, err := p.r.Read(buf)
	if n > 0 {
		p.read += int64(n)
		now := time.Now()
		if err == io.EOF || now.Sub(p.last) >= p.interval {
			p.last = now
			p.onTick(p.read, p.total)
		}
	} else if err == io.EOF {
		p.onTick(p.read, p.total)
	}
	return n, err
}
