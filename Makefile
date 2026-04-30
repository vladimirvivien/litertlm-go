.PHONY: build vet test lint tidy all clean

# Default target: everything CI runs, locally.
all: build vet test lint

build:
	go build ./...

vet:
	go vet ./...

test:
	go test ./...

# Requires golangci-lint on $PATH:
#   go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
lint:
	golangci-lint run ./...

tidy:
	go mod tidy

clean:
	go clean ./...
