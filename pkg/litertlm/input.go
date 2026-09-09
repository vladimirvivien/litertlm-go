package litertlm

import (
	"runtime"
	"unsafe"
)

// Delete releases the allocated InputData handle.
func (i InputData) Delete() { deleteHandle(i, inputDataDeleteFunc) }

func newInputData(t InputDataType, data unsafe.Pointer, size int) (InputData, error) {
	typ := int32(t)
	sz := uint64(size)
	return callStatusHandle[InputData](inputDataCreateFunc, "input_data_create",
		unsafe.Pointer(&typ),
		unsafe.Pointer(&data),
		unsafe.Pointer(&sz),
	)
}

// NewTextInput builds an InputData referencing the UTF-8 bytes of s.
// Since LiteRT-LM copies the data internally on creation, the backing slice
// does not need to be pinned or kept alive after this function returns.
func NewTextInput(s []byte) (InputData, error) {
	return NewBinaryInput(InputText, s)
}

// NewTextInputString is a convenience wrapper over NewTextInput for Go strings.
func NewTextInputString(s string) (InputData, error) {
	var data unsafe.Pointer
	if len(s) > 0 {
		data = unsafe.Pointer(unsafe.StringData(s))
	}
	h, err := newInputData(InputText, data, len(s))
	runtime.KeepAlive(s)
	return h, err
}

// NewBinaryInput builds an InputData for image or audio bytes. Use
// InputImage, InputImageEnd, InputAudio or InputAudioEnd as the type.
func NewBinaryInput(t InputDataType, b []byte) (InputData, error) {
	var data unsafe.Pointer
	if len(b) > 0 {
		data = unsafe.Pointer(&b[0])
	}
	h, err := newInputData(t, data, len(b))
	runtime.KeepAlive(b)
	return h, err
}
