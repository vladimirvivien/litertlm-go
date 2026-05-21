.PHONY: build vet test lint tidy all clean

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

# Requires golangci-lint on $PATH:
#   go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
lint:
	golangci-lint run ./...

tidy:
	go mod tidy

clean:
	go clean ./...
