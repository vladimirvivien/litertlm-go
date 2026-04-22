# Building LiteRT-LM Shared Library

This guide covers building a C-API shared library
(`liblitertlm_c[_cpu].{so,dylib}`) for the [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) 
project from source on **Linux and macOS**.

## Platform support

- **Linux** — tested.
- **macOS** — tested.
- **Windows** — **not tested** with `litertlm-go`.

## 1. Install prerequisites

| Platform   | What to install                                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------------------------------ |
| **Linux**  | `sudo apt install clang git-lfs patchelf` — clang is required (the project's `.bazelrc` pins `CC=clang` on Linux). |
| **macOS**  | `xcode-select --install` (provides clang) and `brew install git-lfs`.                                              |

### Bazel 7.6.1 (exact version)

The repo's `.bazelversion` pins **7.6.1**. 
Use [Bazelisk](https://github.com/bazelbuild/bazelisk) — it reads `.bazelversion` and downloads the matching Bazel automatically:

```bash
# Linux
curl -L -o ~/.local/bin/bazel \
  https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
chmod +x ~/.local/bin/bazel
```

```bash
# macOS
brew install bazelisk
```

---

## 2. Get the source and pull the prebuilt deps

```bash
git clone https://github.com/google-ai-edge/LiteRT-LM.git
cd LiteRT-LM
git lfs install --local
git lfs pull
```

**Verify prebuilt files** 
The `git lfs pull` command downloads pre-built shared libraries for different architectures:

```bash
ls -al prebuilt/
android_arm64  
android_x86_64  
ios_arm64  
ios_sim_arm64
linux_arm64
linux_x86_64
macos_arm64
windows_x86_64
```

Ensure your target environment is listed above.

---

## 3. Build the C-API shared library

Currently, the LiteRT-LM project does not distribute the
C API shared library; it must be built from source. This doc walks
you through creating a Bazel BUILD package that generates the
necessary shared library files.

### 3.1 Create `c/litertlm_c_api/BUILD`

Create a new directory `c/litertlm_c_api/` at the root of your LiteRT-LM
checkout, and put this in `c/litertlm_c_api/BUILD`:

```python
package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "litertlm_c_cpu",
    linkshared = 1,
    deps = ["//c:engine_cpu"],
)

cc_binary(
    name = "litertlm_c",
    linkshared = 1,
    deps = ["//c:engine"],
)
```

### 3.2 Linux

```bash
# CPU-only
bazel build //c/litertlm_c_api:litertlm_c_cpu
# Output: bazel-bin/c/litertlm_c_api/liblitertlm_c_cpu.so

# GPU-capable
bazel build //c/litertlm_c_api:litertlm_c \
    --define=litert_link_capi_so=true \
    --define=resolve_symbols_in_exec=false
# Output: bazel-bin/c/litertlm_c_api/liblitertlm_c.so
```

### 3.3 macOS

```bash
# CPU-only
bazel build //c/litertlm_c_api:litertlm_c_cpu
# Output: bazel-bin/c/litertlm_c_api/liblitertlm_c_cpu.dylib

# GPU-capable
bazel build //c/litertlm_c_api:litertlm_c \
    --define=litert_link_capi_so=true \
    --define=resolve_symbols_in_exec=false
# Output: bazel-bin/c/litertlm_c_api/liblitertlm_c.dylib
```

> The flag `--@litert//litert/build_common:build_include=cpu_only` would
> exclude NPU/GPU registration (silencing the `NPU accelerator could not
> be loaded` runtime warning), but it is currently broken in upstream
> LiteRT — `environment.cc` unconditionally includes a GPU header that
> is gated out of the deps. Until upstream fixes the include guards,
> live with the warning or drop stderr at the OS level (`2>/dev/null`).

### 3.4 Clean rebuild

If a build misbehaves after a config change, wipe the cache and retry:

```bash
bazel clean --expunge
```

---

## 4. Prepare libraries

Next, we need to place all shared libraries in a centralized location 
where they can be loaded at runtime. For instance, for our setup, we
will place all files in `~/include/litertlm/lib`.

### Copy shared libraries files
Let's store the library directory location in environment variable `$LITERTLM_LIB`:

```
export LITERTLM_LIB=~/include/litertlm/lib
```

Copy the prebuilt library files for our target architecture:

```
cp prebuilt/linux_x86_64/*.so $LITERTLM_LIB
```

Next, copy your bazel-built files into the library location as well:

```
cp bazel-bin/c/litertlm_c_api/*.so $LITERTLM_LIB
```

## 5. Using litertlm-go

When you write a program that uses `litertlm-go` you will need to specify
the location of the library files (optionally with the `LITERTLM_LIB`
environment variable) and the model path:

```bash
LITERTLM_LIB=~/include/litertlm/lib \
    go run ./examples/hello -model /path/to/model.litertlm
```

`litertlm-go` handles the platform-specific lib filename mapping
(`liblitertlm_c_cpu.so` on Linux, `liblitertlm_c_cpu.dylib` on macOS).

---

## 6. Troubleshooting

| Symptom                                                                | Cause                                                                                  | Fix                                                                                            |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Link error: *"not an object or archive"* on a `prebuilt/*.so`          | LFS pointer instead of binary                                                          | `git lfs install --local && git lfs pull`                                                      |
| Bazel: *"Cannot find gcc or CC (clang)"* (Linux)                       | clang not installed                                                                    | `sudo apt install clang`                                                                       |
| Bazel: *"requires Bazel 7.6.1 …"*                                      | Wrong Bazel version on PATH                                                            | Install and use Bazelisk (a drop-in replacement)                                               |
| Runtime: *"error while loading shared libraries: libLiteRt.so"*        | GPU plugins missing from `LITERTLM_LIB`                                                | Re-stage the GPU plugins into `LITERTLM_LIB` per §4                                            |
| `engine_create` returns NULL early in setup                            | LFS deps missing or out of date                                                        | Re-run `git lfs pull`; verify file size/`file` output                                          |
| `engine_create` returns NULL with `DYNAMIC_UPDATE_SLICE` error in logs | `max_num_tokens` is smaller than the model's smallest prefill signature (often 128)    | Bump `max_num_tokens` to ≥1024                                                                 |
| `Responses.NumCandidates() == 1` but `Text(0) == ""`                   | Chat-tuned model received raw text without its template                                | Use the **Conversation** API instead of `Session.GenerateContent`, or wrap the prompt manually |

