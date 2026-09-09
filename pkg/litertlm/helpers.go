package litertlm

import (
	"github.com/vladimirvivien/litertlm-go/internal/cstr"
)

// bytePtrOrNil returns a null-terminated byte pointer for s, or nil for
// the empty string. Several C entry points accept NULL for "unset".
func bytePtrOrNil(s string) (*byte, error) {
	if s == "" {
		return nil, nil
	}
	return cstr.BytePtrFromString(s)
}
