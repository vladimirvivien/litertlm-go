package litertlm

import (
	"unsafe"

	"github.com/vladimirvivien/litertlm-go/pkg/utils"
)

// Delete releases a Responses handle.
func (r Responses) Delete() {
	if r == 0 {
		return
	}
	responsesDeleteFunc.Call(nil, unsafe.Pointer(&r))
}

// NumCandidates returns the number of response candidates available.
func (r Responses) NumCandidates() int {
	if r == 0 {
		return 0
	}
	var n int32
	responsesGetNumCandidatesFunc.Call(unsafe.Pointer(&n), unsafe.Pointer(&r))
	return int(n)
}

// Text returns the response text at index i, copied into Go memory.
// Returns "" if the index is out of bounds or the underlying pointer is
// null. The copy lets the result outlive Responses.Delete().
func (r Responses) Text(i int) string {
	if r == 0 {
		return ""
	}
	var ptr *byte
	responsesGetResponseTextAtFunc.Call(
		unsafe.Pointer(&ptr),
		unsafe.Pointer(&r),
		unsafe.Pointer(new(int32(i))),
	)
	if ptr == nil {
		return ""
	}
	return utils.BytePtrToString(ptr)
}
