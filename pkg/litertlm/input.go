package litertlm

import (
	"fmt"
	"unsafe"
)

// Delete releases the allocated InputData handle.
func (i InputData) Delete() {
	if i == 0 {
		return
	}
	inputDataDeleteFunc.Call(nil, unsafe.Pointer(&i))
}

// NewTextInput builds an InputData referencing the UTF-8 bytes of s.
// Since LiteRT-LM copies the data internally on creation, the backing slice
// does not need to be pinned or kept alive after this function returns.
func NewTextInput(s []byte) (InputData, error) {
	var data unsafe.Pointer
	if len(s) > 0 {
		data = unsafe.Pointer(&s[0])
	}
	var h InputData
	t := int32(InputText)
	size := uint64(len(s))
	inputDataCreateFunc.Call(
		unsafe.Pointer(&h),
		unsafe.Pointer(&t),
		unsafe.Pointer(&data),
		unsafe.Pointer(&size),
	)
	if h == 0 {
		return 0, fmt.Errorf("litertlm: input_data_create failed for text input")
	}
	return h, nil
}

// NewTextInputString is a convenience wrapper over NewTextInput for Go
// strings.
func NewTextInputString(s string) (InputData, error) {
	var data unsafe.Pointer
	if len(s) > 0 {
		data = unsafe.Pointer(unsafe.StringData(s))
	}
	var h InputData
	t := int32(InputText)
	size := uint64(len(s))
	inputDataCreateFunc.Call(
		unsafe.Pointer(&h),
		unsafe.Pointer(&t),
		unsafe.Pointer(&data),
		unsafe.Pointer(&size),
	)
	if h == 0 {
		return 0, fmt.Errorf("litertlm: input_data_create failed for text input string")
	}
	return h, nil
}

// NewBinaryInput builds an InputData for image or audio bytes. Use
// InputImage, InputImageEnd, InputAudio or InputAudioEnd as the type.
func NewBinaryInput(t InputDataType, b []byte) (InputData, error) {
	var data unsafe.Pointer
	if len(b) > 0 {
		data = unsafe.Pointer(&b[0])
	}
	var h InputData
	typ := int32(t)
	size := uint64(len(b))
	inputDataCreateFunc.Call(
		unsafe.Pointer(&h),
		unsafe.Pointer(&typ),
		unsafe.Pointer(&data),
		unsafe.Pointer(&size),
	)
	if h == 0 {
		return 0, fmt.Errorf("litertlm: input_data_create failed for binary input type %d", t)
	}
	return h, nil
}
