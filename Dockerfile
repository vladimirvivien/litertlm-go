# syntax=docker/dockerfile:1.7
#
# Builds the LiteRT-LM C shared libraries and extracts them into a local
# directory ready to use as LITERTLM_LIB. Linux x86_64 only.
#
#   docker build --target cpu -o $LITERTLM_LIB .
#   docker build --target gpu -o $LITERTLM_LIB .
#
# The GPU variant links the GPU-capable C API; it still needs working Vulkan
# drivers on the host to actually run.

FROM debian:trixie-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates clang curl git git-lfs python3 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /usr/local/bin/bazel \
        https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
    && chmod +x /usr/local/bin/bazel

WORKDIR /src
RUN git clone --depth=1 https://github.com/google-ai-edge/LiteRT-LM.git . \
    && git lfs install --local \
    && git lfs pull

RUN mkdir -p c/litertlm_c_api && cat > c/litertlm_c_api/BUILD <<'EOF'
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
EOF

FROM base AS builder-cpu
RUN --mount=type=cache,target=/root/.cache/bazel \
    set -eux; \
    bazel build //c/litertlm_c_api:litertlm_c_cpu; \
    mkdir -p /out; \
    cp prebuilt/linux_x86_64/*.so /out/; \
    cp -L bazel-bin/c/litertlm_c_api/liblitertlm_c_cpu.so /out/

FROM base AS builder-gpu
RUN --mount=type=cache,target=/root/.cache/bazel \
    set -eux; \
    bazel build //c/litertlm_c_api:litertlm_c \
        --define=litert_link_capi_so=true \
        --define=resolve_symbols_in_exec=false; \
    mkdir -p /out; \
    cp prebuilt/linux_x86_64/*.so /out/; \
    cp -L bazel-bin/c/litertlm_c_api/liblitertlm_c.so /out/

FROM scratch AS cpu
COPY --from=builder-cpu /out/ /

FROM scratch AS gpu
COPY --from=builder-gpu /out/ /
