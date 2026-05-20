.PHONY: build vet test lint tidy all clean

# Default target: everything CI runs, locally.
all: build vet test lint

build:
	go build ./...

vet:
	go vet ./...

test:
	go test ./...

# Integration tests in pkg/litertlm/ exercise a real .litertlm model
# and only run when the following env vars are set; otherwise they
# t.Skip:
#   LITERTLM_TEST_LIB   = staging dir holding the C-API DLLs
#   LITERTLM_TEST_MODEL = absolute path to a Gemma 4 .litertlm file
# Run them via the standard `test` target with both vars exported.

# Requires golangci-lint on $PATH:
#   go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
lint:
	golangci-lint run ./...

tidy:
	go mod tidy

clean:
	go clean ./...
