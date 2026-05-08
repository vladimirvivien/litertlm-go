package litertlm

import (
	"time"
	"unsafe"
)

// Benchmark is a pure-Go snapshot of a BenchmarkInfo handle. Time
// fields use time.Duration; per-turn slices are sized from the
// recorded turn counts.
type Benchmark struct {
	TimeToFirstToken    time.Duration
	TotalInitTime       time.Duration
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
		PrefillTurns:        prefillTurns,
		DecodeTurns:         decodeTurns,
		PrefillTokenCounts:  make([]int, prefillTurns),
		DecodeTokenCounts:   make([]int, decodeTurns),
		PrefillTokensPerSec: make([]float64, prefillTurns),
		DecodeTokensPerSec:  make([]float64, decodeTurns),
	}
	for i := 0; i < prefillTurns; i++ {
		b.PrefillTokenCounts[i] = h.PrefillTokenCount(i)
		b.PrefillTokensPerSec[i] = h.PrefillTokensPerSec(i)
	}
	for i := 0; i < decodeTurns; i++ {
		b.DecodeTokenCounts[i] = h.DecodeTokenCount(i)
		b.DecodeTokensPerSec[i] = h.DecodeTokensPerSec(i)
	}
	return b
}

func seconds(s float64) time.Duration {
	return time.Duration(s * float64(time.Second))
}

// Delete releases a BenchmarkInfo handle.
func (b BenchmarkInfo) Delete() {
	if b == 0 {
		return
	}
	benchmarkInfoDeleteFunc.Call(nil, unsafe.Pointer(&b))
}

// TimeToFirstToken returns the prefill + first decode time, in seconds. Per
// the C-header contract, this does NOT include one-time initialisation.
func (b BenchmarkInfo) TimeToFirstToken() float64 {
	if b == 0 {
		return 0
	}
	var v float64
	benchmarkInfoGetTimeToFirstTokenFunc.Call(unsafe.Pointer(&v), unsafe.Pointer(&b))
	return v
}

// TotalInitTime returns total engine initialisation time, in seconds.
func (b BenchmarkInfo) TotalInitTime() float64 {
	if b == 0 {
		return 0
	}
	var v float64
	benchmarkInfoGetTotalInitTimeInSecondFunc.Call(unsafe.Pointer(&v), unsafe.Pointer(&b))
	return v
}

// NumPrefillTurns returns the number of recorded prefill turns.
func (b BenchmarkInfo) NumPrefillTurns() int {
	if b == 0 {
		return 0
	}
	var v int32
	benchmarkInfoGetNumPrefillTurnsFunc.Call(unsafe.Pointer(&v), unsafe.Pointer(&b))
	return int(v)
}

// NumDecodeTurns returns the number of recorded decode turns.
func (b BenchmarkInfo) NumDecodeTurns() int {
	if b == 0 {
		return 0
	}
	var v int32
	benchmarkInfoGetNumDecodeTurnsFunc.Call(unsafe.Pointer(&v), unsafe.Pointer(&b))
	return int(v)
}

// PrefillTokenCount returns the prefill token count at turn i.
func (b BenchmarkInfo) PrefillTokenCount(i int) int {
	if b == 0 {
		return 0
	}
	var v int32
	benchmarkInfoGetPrefillTokenCountAtFunc.Call(
		unsafe.Pointer(&v),
		unsafe.Pointer(&b),
		unsafe.Pointer(new(int32(i))),
	)
	return int(v)
}

// DecodeTokenCount returns the decode token count at turn i.
func (b BenchmarkInfo) DecodeTokenCount(i int) int {
	if b == 0 {
		return 0
	}
	var v int32
	benchmarkInfoGetDecodeTokenCountAtFunc.Call(
		unsafe.Pointer(&v),
		unsafe.Pointer(&b),
		unsafe.Pointer(new(int32(i))),
	)
	return int(v)
}

// PrefillTokensPerSec returns prefill throughput at turn i, in tokens/sec.
func (b BenchmarkInfo) PrefillTokensPerSec(i int) float64 {
	if b == 0 {
		return 0
	}
	var v float64
	benchmarkInfoGetPrefillTokensPerSecAtFunc.Call(
		unsafe.Pointer(&v),
		unsafe.Pointer(&b),
		unsafe.Pointer(new(int32(i))),
	)
	return v
}

// DecodeTokensPerSec returns decode throughput at turn i, in tokens/sec.
func (b BenchmarkInfo) DecodeTokensPerSec(i int) float64 {
	if b == 0 {
		return 0
	}
	var v float64
	benchmarkInfoGetDecodeTokensPerSecAtFunc.Call(
		unsafe.Pointer(&v),
		unsafe.Pointer(&b),
		unsafe.Pointer(new(int32(i))),
	)
	return v
}
