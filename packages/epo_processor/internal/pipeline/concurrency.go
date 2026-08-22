package pipeline

// PlanReaderConcurrency resolves the number of concurrent XML readers
// (in-flight entries) to run.
//
// Each reader tokenizes one entry sequentially and is CPU-bound, so the
// number of readers is what saturates cores. An explicit positive request
// always wins. When requested <= 0 ("auto"), it auto-sizes to numCPU but
// clamps by a memory budget, since every in-flight reader buffers a
// decompressed XML stream plus the current DOM node (~perEntryBytes).
//
// The result is always at least 1.
func PlanReaderConcurrency(requested, numCPU int, memBudgetBytes, perEntryBytes int64) int {
	if requested > 0 {
		return requested
	}

	n := numCPU
	if n < 1 {
		n = 1
	}

	if perEntryBytes > 0 && memBudgetBytes > 0 {
		if cap := int(memBudgetBytes / perEntryBytes); cap < n {
			n = cap
		}
	}

	if n < 1 {
		n = 1
	}
	return n
}
