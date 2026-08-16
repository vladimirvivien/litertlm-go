//go:build windows

package loader

import (
	"syscall"
	"unsafe"
)

var setDllDirectory = syscall.NewLazyDLL("kernel32.dll").NewProc("SetDllDirectoryW")

func setDllDir(path string) {
	if path == "" {
		return
	}
	p, err := syscall.UTF16PtrFromString(path)
	if err == nil {
		_, _, _ = setDllDirectory.Call(uintptr(unsafe.Pointer(p)))
	}
}
