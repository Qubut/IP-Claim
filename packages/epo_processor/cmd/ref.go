package cmd

import (
	"context"
	"sync"

	IOG "github.com/IBM/fp-go/v2/io"
)

// ref is a concurrency-safe mutable cell.
type ref[A any] struct {
	mu sync.Mutex
	v  A
}

// newRef builds a cell holding initial.
func newRef[A any](initial A) *ref[A] { return &ref[A]{v: initial} }

// acquire is the lock-acquiring IO consumed by io.WithLock: it locks and
// returns the matching release as a context.CancelFunc.
func (r *ref[A]) acquire() IOG.IO[context.CancelFunc] {
	return func() context.CancelFunc {
		r.mu.Lock()
		return r.mu.Unlock
	}
}

// Set returns an IO that stores v under the lock and yields it.
func (r *ref[A]) Set(v A) IOG.IO[A] {
	return IOG.WithLock[A](r.acquire())(func() A {
		r.v = v
		return v
	})
}

// Get returns an IO that reads the current value under the lock.
func (r *ref[A]) Get() IOG.IO[A] {
	return IOG.WithLock[A](r.acquire())(func() A { return r.v })
}
