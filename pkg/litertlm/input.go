package litertlm

import "unsafe"

// InputData is the Go representation of the C `InputData` struct from
// c/engine.h. Layout must match the C struct exactly:
//
//	typedef struct {
//	    InputDataType type;   // int32 enum, 4 bytes
//	    const void*   data;   // pointer, 8 bytes (after 4-byte pad)
//	    size_t        size;   // 8 bytes
//	} InputData;
//
// Total 24 bytes on 64-bit with natural alignment — identical to the Go
// struct below.
type InputData struct {
	Type InputDataType
	_    [4]byte // explicit pad so the layout is obvious to readers
	Data unsafe.Pointer
	Size uintptr
}

// NewTextInput builds an InputData that references the UTF-8 bytes of s.
// The returned record is only valid for the lifetime of the supplied slice —
// callers are responsible for keeping a reference alive across the C call
// (typically by storing the slice in a local variable that outlives the
// GenerateContent / GenerateContentStream invocation).
func NewTextInput(s []byte) InputData {
	var data unsafe.Pointer
	if len(s) > 0 {
		data = unsafe.Pointer(&s[0])
	}
	return InputData{
		Type: InputText,
		Data: data,
		Size: uintptr(len(s)),
	}
}

// NewTextInputString is a convenience wrapper over NewTextInput for Go
// strings. The string's backing bytes are referenced directly via unsafe —
// valid because Go strings are immutable for the duration of the call.
func NewTextInputString(s string) InputData {
	var data unsafe.Pointer
	if len(s) > 0 {
		data = unsafe.Pointer(unsafe.StringData(s))
	}
	return InputData{
		Type: InputText,
		Data: data,
		Size: uintptr(len(s)),
	}
}

// NewBinaryInput builds an InputData for image or audio bytes. Use
// InputImage, InputImageEnd, InputAudio or InputAudioEnd as the type.
func NewBinaryInput(t InputDataType, b []byte) InputData {
	var data unsafe.Pointer
	if len(b) > 0 {
		data = unsafe.Pointer(&b[0])
	}
	return InputData{
		Type: t,
		Data: data,
		Size: uintptr(len(b)),
	}
}
