package litertlm

import "unsafe"

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
	idx := int32(i)
	var v int32
	benchmarkInfoGetPrefillTokenCountAtFunc.Call(
		unsafe.Pointer(&v),
		unsafe.Pointer(&b),
		unsafe.Pointer(&idx),
	)
	return int(v)
}

// DecodeTokenCount returns the decode token count at turn i.
func (b BenchmarkInfo) DecodeTokenCount(i int) int {
	if b == 0 {
		return 0
	}
	idx := int32(i)
	var v int32
	benchmarkInfoGetDecodeTokenCountAtFunc.Call(
		unsafe.Pointer(&v),
		unsafe.Pointer(&b),
		unsafe.Pointer(&idx),
	)
	return int(v)
}

// PrefillTokensPerSec returns prefill throughput at turn i, in tokens/sec.
func (b BenchmarkInfo) PrefillTokensPerSec(i int) float64 {
	if b == 0 {
		return 0
	}
	idx := int32(i)
	var v float64
	benchmarkInfoGetPrefillTokensPerSecAtFunc.Call(
		unsafe.Pointer(&v),
		unsafe.Pointer(&b),
		unsafe.Pointer(&idx),
	)
	return v
}

// DecodeTokensPerSec returns decode throughput at turn i, in tokens/sec.
func (b BenchmarkInfo) DecodeTokensPerSec(i int) float64 {
	if b == 0 {
		return 0
	}
	idx := int32(i)
	var v float64
	benchmarkInfoGetDecodeTokensPerSecAtFunc.Call(
		unsafe.Pointer(&v),
		unsafe.Pointer(&b),
		unsafe.Pointer(&idx),
	)
	return v
}
