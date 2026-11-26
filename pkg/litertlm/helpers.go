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

func marshalBackendPtrs(backend string, vision, audio *string) (backendPtr, visionPtr, audioPtr *byte, err error) {
	backendPtr, err = cstr.BytePtrFromString(backend)
	if err != nil {
		return nil, nil, nil, err
	}
	if vision != nil {
		visionPtr, err = cstr.BytePtrFromString(*vision)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	if audio != nil {
		audioPtr, err = cstr.BytePtrFromString(*audio)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	return backendPtr, visionPtr, audioPtr, nil
}
