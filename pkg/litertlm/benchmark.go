package litertlm

import "time"

// Benchmark is a pure-Go snapshot of a BenchmarkInfo handle. Time
// fields use time.Duration; per-turn slices are sized from the
// recorded turn counts.
type Benchmark struct {
	TimeToFirstToken    time.Duration
	TotalInitTime       time.Duration
	MaxNumTokens        int
	PrefillTurns        int
	DecodeTurns         int
	PrefillTokenCounts  []int
	DecodeTokenCounts   []int
	PrefillTokensPerSec []float64
	DecodeTokensPerSec  []float64
}

// snapshotBenchmark walks h's getters and returns a pure-Go copy.
// Caller still owns h and must Delete() it. Returns nil for the
// zero handle.
func snapshotBenchmark(h BenchmarkInfo) *Benchmark {
	if h == 0 {
		return nil
	}
	prefillTurns := h.NumPrefillTurns()
	decodeTurns := h.NumDecodeTurns()

	b := &Benchmark{
		TimeToFirstToken:    seconds(h.TimeToFirstToken()),
		TotalInitTime:       seconds(h.TotalInitTime()),
		MaxNumTokens:        h.MaxNumTokens(),
		PrefillTurns:        prefillTurns,
		DecodeTurns:         decodeTurns,
		PrefillTokenCounts:  make([]int, prefillTurns),
		DecodeTokenCounts:   make([]int, decodeTurns),
		PrefillTokensPerSec: make([]float64, prefillTurns),
		DecodeTokensPerSec:  make([]float64, decodeTurns),
	}
	for i := range prefillTurns {
		b.PrefillTokenCounts[i] = h.PrefillTokenCount(i)
		b.PrefillTokensPerSec[i] = h.PrefillTokensPerSec(i)
	}
	for i := range decodeTurns {
		b.DecodeTokenCounts[i] = h.DecodeTokenCount(i)
		b.DecodeTokensPerSec[i] = h.DecodeTokensPerSec(i)
	}
	return b
}

func seconds(s float64) time.Duration {
	return time.Duration(s * float64(time.Second))
}

// Delete releases a BenchmarkInfo handle.
func (b BenchmarkInfo) Delete() { deleteHandle(b, benchmarkInfoDeleteFunc) }

// TimeToFirstToken returns the prefill + first decode time, in seconds. Per
// the C-header contract, this does NOT include one-time initialisation.
func (b BenchmarkInfo) TimeToFirstToken() float64 {
	return getHandleScalar[BenchmarkInfo, float64](b, benchmarkInfoGetTimeToFirstTokenFunc, "benchmark_info_get_time_to_first_token", 0)
}

// TotalInitTime returns engine initialisation time, in seconds.
func (b BenchmarkInfo) TotalInitTime() float64 {
	return getHandleScalar[BenchmarkInfo, float64](b, benchmarkInfoGetTotalInitTimeInSecondFunc, "benchmark_info_get_total_init_time_in_second", 0)
}

// NumPrefillTurns returns the number of recorded prefill turns.
func (b BenchmarkInfo) NumPrefillTurns() int {
	return int(getHandleScalar[BenchmarkInfo, uint64](b, benchmarkInfoGetNumPrefillTurnsFunc, "benchmark_info_get_num_prefill_turns", 0))
}

// NumDecodeTurns returns the number of recorded decode turns.
func (b BenchmarkInfo) NumDecodeTurns() int {
	return int(getHandleScalar[BenchmarkInfo, uint64](b, benchmarkInfoGetNumDecodeTurnsFunc, "benchmark_info_get_num_decode_turns", 0))
}

// PrefillTokenCount returns the prefill token count at turn i.
func (b BenchmarkInfo) PrefillTokenCount(i int) int {
	return int(getHandleScalarAt[BenchmarkInfo, uint64](b, i, benchmarkInfoGetPrefillTokenCountAtFunc, "benchmark_info_get_prefill_token_count_at", 0))
}

// DecodeTokenCount returns the decode token count at turn i.
func (b BenchmarkInfo) DecodeTokenCount(i int) int {
	return int(getHandleScalarAt[BenchmarkInfo, uint64](b, i, benchmarkInfoGetDecodeTokenCountAtFunc, "benchmark_info_get_decode_token_count_at", 0))
}

// PrefillTokensPerSec returns prefill throughput at turn i, in tokens/sec.
func (b BenchmarkInfo) PrefillTokensPerSec(i int) float64 {
	return getHandleScalarAt[BenchmarkInfo, float64](b, i, benchmarkInfoGetPrefillTokensPerSecAtFunc, "benchmark_info_get_prefill_tokens_per_sec_at", 0)
}

// DecodeTokensPerSec returns decode throughput at turn i, in tokens/sec.
func (b BenchmarkInfo) DecodeTokensPerSec(i int) float64 {
	return getHandleScalarAt[BenchmarkInfo, float64](b, i, benchmarkInfoGetDecodeTokensPerSecAtFunc, "benchmark_info_get_decode_tokens_per_sec_at", 0)
}

// MaxNumTokens returns the KV cache token capacity reported by the engine.
func (b BenchmarkInfo) MaxNumTokens() int {
	return int(getHandleScalar[BenchmarkInfo, int32](b, benchmarkInfoGetMaxNumTokensFunc, "benchmark_info_get_max_num_tokens", 0))
}
