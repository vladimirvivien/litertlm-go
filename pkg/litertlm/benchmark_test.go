package litertlm

import (
	"testing"
	"time"
)

func TestSeconds(t *testing.T) {
	cases := []struct {
		in   float64
		want time.Duration
	}{
		{0, 0},
		{1, time.Second},
		{0.5, 500 * time.Millisecond},
		{0.001, time.Millisecond},
		{2.5, 2500 * time.Millisecond},
	}
	for _, tc := range cases {
		if got := seconds(tc.in); got != tc.want {
			t.Errorf("seconds(%v) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

func TestSnapshotBenchmark_ZeroHandle(t *testing.T) {
	if got := snapshotBenchmark(BenchmarkInfo(0)); got != nil {
		t.Errorf("snapshot of zero handle = %v, want nil", got)
	}
}

func TestBenchmarkStruct_PerTurnSlicesAlignWithCounts(t *testing.T) {
	// Fixture mimicking what snapshotBenchmark would build: per-turn
	// slices sized to PrefillTurns / DecodeTurns and aligned by index.
	b := &Benchmark{
		TimeToFirstToken:    150 * time.Millisecond,
		TotalInitTime:       2 * time.Second,
		PrefillTurns:        2,
		DecodeTurns:         3,
		PrefillTokenCounts:  []int{16, 32},
		DecodeTokenCounts:   []int{50, 75, 100},
		PrefillTokensPerSec: []float64{300.0, 305.0},
		DecodeTokensPerSec:  []float64{20.0, 21.0, 22.0},
	}
	if len(b.PrefillTokenCounts) != b.PrefillTurns {
		t.Errorf("PrefillTokenCounts len = %d, want %d", len(b.PrefillTokenCounts), b.PrefillTurns)
	}
	if len(b.DecodeTokensPerSec) != b.DecodeTurns {
		t.Errorf("DecodeTokensPerSec len = %d, want %d", len(b.DecodeTokensPerSec), b.DecodeTurns)
	}
}

func TestResponse_Benchmark(t *testing.T) {
	r := &Response{}
	if r.Benchmark() != nil {
		t.Errorf("default Benchmark() = %v, want nil", r.Benchmark())
	}
	want := &Benchmark{TotalInitTime: time.Second}
	r.bench = want
	if r.Benchmark() != want {
		t.Errorf("Benchmark() did not return the assigned value")
	}
	var nilResp *Response
	if nilResp.Benchmark() != nil {
		t.Errorf("nil Response Benchmark() should be nil")
	}
}
