package pipeline

import "testing"

const (
	gb = int64(1) << 30
	mb = int64(1) << 20
)

func TestPlanReaderConcurrency(t *testing.T) {
	tests := []struct {
		name           string
		requested      int
		numCPU         int
		memBudgetBytes int64
		perEntryBytes  int64
		want           int
	}{
		{"explicit request wins over auto-sizing", 16, 64, 32 * gb, 256 * mb, 16},
		{"explicit request ignores memory budget", 200, 8, 1 * gb, 256 * mb, 200},
		{"auto sizes to numCPU when budget is ample", 0, 64, 1024 * gb, 256 * mb, 64},
		{"auto clamps to memory budget below numCPU", 0, 64, 4 * gb, 256 * mb, 16},
		{"auto clamps to 1 when budget is tiny", 0, 64, 100 * mb, 256 * mb, 1},
		{"auto without budget falls back to numCPU", 0, 32, 0, 0, 32},
		{"auto without per-entry estimate falls back to numCPU", 0, 32, 32 * gb, 0, 32},
		{"never returns less than 1", 0, 0, 0, 0, 1},
		{"negative request treated as auto", -1, 8, 1024 * gb, 256 * mb, 8},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := PlanReaderConcurrency(tt.requested, tt.numCPU, tt.memBudgetBytes, tt.perEntryBytes)
			if got != tt.want {
				t.Errorf("PlanReaderConcurrency(%d, %d, %d, %d) = %d, want %d",
					tt.requested, tt.numCPU, tt.memBudgetBytes, tt.perEntryBytes, got, tt.want)
			}
		})
	}
}
