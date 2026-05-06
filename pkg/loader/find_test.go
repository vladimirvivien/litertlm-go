package loader

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// touchLib creates an empty file at GetLibraryFilename(dir, libName)
// so Find sees the candidate as resolved.
func touchLib(t *testing.T, dir, libName string) string {
	t.Helper()
	full := GetLibraryFilename(dir, libName)
	if err := os.WriteFile(full, []byte{}, 0o600); err != nil {
		t.Fatalf("write fixture lib: %v", err)
	}
	return full
}

// TestFind_FirstHit returns the earliest candidate where the lib
// exists.
func TestFind_FirstHit(t *testing.T) {
	dir1 := t.TempDir()
	dir2 := t.TempDir()
	touchLib(t, dir2, "litertlm_c_cpu")

	got, err := Find("litertlm_c_cpu", []string{dir1, dir2})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if got != dir2 {
		t.Errorf("got %q, want %q", got, dir2)
	}
}

// TestFind_AllMiss returns an error that names every checked path so
// the user can see exactly where Find looked.
func TestFind_AllMiss(t *testing.T) {
	dir1 := t.TempDir()
	dir2 := t.TempDir()
	_, err := Find("litertlm_c_cpu", []string{dir1, dir2})
	if err == nil {
		t.Fatal("expected error on all-miss")
	}
	for _, dir := range []string{dir1, dir2} {
		want := GetLibraryFilename(dir, "litertlm_c_cpu")
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not name candidate %q", err, want)
		}
	}
}

// TestFind_SkipsEmptyCandidates tolerates "" entries (callers compose
// the list from possibly-unset sources like an empty WithLib option).
func TestFind_SkipsEmptyCandidates(t *testing.T) {
	dir := t.TempDir()
	touchLib(t, dir, "litertlm_c_cpu")
	got, err := Find("litertlm_c_cpu", []string{"", dir, ""})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if got != dir {
		t.Errorf("got %q, want %q", got, dir)
	}
}

// TestFind_NoCandidates returns a distinguishable error.
func TestFind_NoCandidates(t *testing.T) {
	_, err := Find("litertlm_c_cpu", nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "no candidate directories") {
		t.Errorf("error %q does not mention missing candidates", err)
	}
}

// TestFind_RequiresLibName guards against accidentally calling Find
// with no name.
func TestFind_RequiresLibName(t *testing.T) {
	_, err := Find("", []string{t.TempDir()})
	if err == nil {
		t.Fatal("expected error on empty libName")
	}
}

// TestFind_ExpandsTilde resolves a "~/..." candidate against
// os.UserHomeDir before checking.
func TestFind_ExpandsTilde(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	if runtime.GOOS == "windows" {
		t.Setenv("USERPROFILE", home)
	}
	subdir := filepath.Join(home, "litertlm-fixture")
	if err := os.MkdirAll(subdir, 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	touchLib(t, subdir, "litertlm_c_cpu")

	got, err := Find("litertlm_c_cpu", []string{"~/litertlm-fixture"})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if got != subdir {
		t.Errorf("got %q, want %q", got, subdir)
	}
}

// TestDefaultPaths returns a non-empty list on every supported
// platform; sanity check the well-known entries.
func TestDefaultPaths(t *testing.T) {
	paths := DefaultPaths()
	if len(paths) == 0 {
		t.Fatal("DefaultPaths returned empty")
	}
	wantContains := map[string]string{
		"linux":   "/usr/local/lib",
		"darwin":  "/usr/local/lib",
		"freebsd": "/usr/local/lib",
		"windows": "litertlm",
	}
	if want, ok := wantContains[runtime.GOOS]; ok {
		var found bool
		for _, p := range paths {
			if strings.Contains(p, want) {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("DefaultPaths %v does not contain %q", paths, want)
		}
	}
}
