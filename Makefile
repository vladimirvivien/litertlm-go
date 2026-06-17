.PHONY: build vet test lint tidy all clean engdiff evals

# Default target: everything CI runs, locally.
all: build vet test lint

build:
	go build ./...

vet:
	go vet ./...

test:
	go test ./...

# Integration tests in pkg/litertlm/ exercise a real .litertlm model.
# They skip when testing.Short() is set or when either env var below
# is unset. `go test ./...` without env vars and `go test -short ./...`
# both skip them.
#
# To run them, leave -short off and set:
#   LITERTLM_TEST_LIB   = staging dir holding the C-API DLLs
#   LITERTLM_TEST_MODEL = absolute path to a .litertlm file

# Cross-engine regression gates (tools/engdiff). Both need:
#   LITERTLM_LIB = staging dir holding the C-API DLLs
#   LITERT_LIB   = LiteRT runtime dir (libfetch prints it)
#   MODELS       = .litertlm file, comma-separated list, or directory
#
# engdiff is Tier 1 (byte-exact greedy diff per model); evals is
# Tier 2/3 (task accuracy with a cross-engine delta gate + per-engine
# perf records; set PERF_OUT / PERF_BASELINE to record or gate).
engdiff:
	go run ./tools/engdiff -models "$(MODELS)"

evals:
	go run ./tools/engdiff -models "$(MODELS)" -eval tools/engdiff/testdata/evalset.json \
		$(if $(PERF_OUT),-perf-out "$(PERF_OUT)") $(if $(PERF_BASELINE),-perf-baseline "$(PERF_BASELINE)")

# Requires golangci-lint on $PATH:
#   go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
lint:
	golangci-lint run ./...

tidy:
	go mod tidy

clean:
	go clean ./...
