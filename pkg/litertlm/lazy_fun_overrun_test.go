package litertlm

import (
	"encoding/binary"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/jupiterrider/ffi"
)

// TestLazyFun_NarrowReturnDoesNotOverrun is the deterministic guard for
// the FFI return-slot heap-corruption bug. It does not need a model or
// the LiteRT-LM library: a purego callback stands in for a C function
// returning a sub-register integral, and libffi widens that return to a
// full ffi_arg (8 bytes) and writes that many bytes through the return
// pointer. A guard byte is placed immediately after the narrow return
// slot, exactly where the widened write spills. Before the fix
// (lazyFun.Call forwarding the caller's narrow slot straight to libffi)
// the guard is clobbered; with the fix (route the return through a
// register-width buffer, copy back exactly the declared width) the
// guard survives.
//
// The "overrun" stays within a single 8-byte array, so the test detects
// the defect without risking real memory corruption.
func TestLazyFun_NarrowReturnDoesNotOverrun(t *testing.T) {
	tests := []struct {
		name    string
		retType *ffi.Type // declared C return type (drives libffi widening)
		value   uint64    // value the stand-in C function returns
	}{
		{"sint32", &ffi.TypeSint32, 0x11223344},
		{"uint8", &ffi.TypeUint8, 0x05},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// A nullary C function returning the value in a register.
			value := tc.value
			cb := purego.NewCallback(func() uintptr { return uintptr(value) })

			var cif ffi.Cif
			if st := ffi.PrepCif(&cif, ffi.DefaultAbi, 0, tc.retType); st != ffi.OK {
				t.Fatalf("PrepCif: %v", st)
			}
			fun := ffi.Fun{Addr: cb, Cif: &cif}

			withStubPrep(t, func(string, *ffi.Type, ...*ffi.Type) (ffi.Fun, error) {
				return fun, nil
			})
			lf := newLazyFun("stub_narrow_return", tc.retType)

			// buf[0:size] is the caller's return slot; buf[size:] is the
			// guard, pre-filled with a non-zero pattern so a spilled
			// (zero-high) widened write is detectable.
			size := int(tc.retType.Size)
			var buf [8]byte
			for i := range buf {
				buf[i] = 0xAA
			}
			lf.Call(unsafe.Pointer(&buf[0]))

			// The declared-width return value must arrive intact.
			var got uint64
			switch size {
			case 1:
				got = uint64(buf[0])
			case 4:
				got = uint64(binary.LittleEndian.Uint32(buf[0:4]))
			}
			if got != tc.value {
				t.Errorf("return value = %#x, want %#x", got, tc.value)
			}

			// Every byte past the declared width must be untouched.
			for i := size; i < len(buf); i++ {
				if buf[i] != 0xAA {
					t.Fatalf("guard byte %d = %#x, want 0xAA: lazyFun.Call overran the %d-byte "+
						"return slot (libffi ret-width overrun is back)", i, buf[i], size)
				}
			}
		})
	}
}
