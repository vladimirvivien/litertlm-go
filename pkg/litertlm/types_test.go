package litertlm

import (
	"testing"
	"unsafe"
)

// All handle types must alias uintptr so the FFI layer can pass them as
// pointer-sized integers. Their zero value must be 0 (null) so the
// `if h == 0` null check used throughout the package works.
func TestHandleTypesAreUintptr(t *testing.T) {
	want := unsafe.Sizeof(uintptr(0))

	checks := []struct {
		name string
		size uintptr
		zero uintptr
	}{
		{"Engine", unsafe.Sizeof(Engine(0)), uintptr(Engine(0))},
		{"EngineSettings", unsafe.Sizeof(EngineSettings(0)), uintptr(EngineSettings(0))},
		{"Session", unsafe.Sizeof(Session(0)), uintptr(Session(0))},
		{"SessionConfig", unsafe.Sizeof(SessionConfig(0)), uintptr(SessionConfig(0))},
		{"Responses", unsafe.Sizeof(Responses(0)), uintptr(Responses(0))},
		{"BenchmarkInfo", unsafe.Sizeof(BenchmarkInfo(0)), uintptr(BenchmarkInfo(0))},
		{"Conversation", unsafe.Sizeof(Conversation(0)), uintptr(Conversation(0))},
		{"ConversationConfig", unsafe.Sizeof(ConversationConfig(0)), uintptr(ConversationConfig(0))},
		{"JsonResponse", unsafe.Sizeof(JsonResponse(0)), uintptr(JsonResponse(0))},
	}
	for _, c := range checks {
		t.Run(c.name, func(t *testing.T) {
			if c.size != want {
				t.Errorf("sizeof(%s) = %d, want %d", c.name, c.size, want)
			}
			if c.zero != 0 {
				t.Errorf("zero value of %s = %d, want 0", c.name, c.zero)
			}
		})
	}
}
