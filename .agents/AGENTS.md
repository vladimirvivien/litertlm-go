# Custom Rules

- Use `jj` instead of `git` for all version control operations (status, diff, commit, etc.).
- Never mutate or create commits unless explicitly directed by the user. Keep work in the working copy `@`.

## 1. General Documentation Tone & Voice
* **Style Constraints**: Documentation must adopt an instructive, direct, developer-to-developer, and concise tone. 
* **Hype Avoidance**: Avoid generic marketing adjectives, superlatives, and buzzwords (e.g., "robust", "powerful", "seamless", "awesome", "revolutionary").

## 2. Quickstart & Command Path Consistency
* **Local Context**: In hello-world or quickstart tutorials where the reader is instructed to create a script file locally, all subsequent shell commands showing execution output must run against the local file path (e.g., `./hello.star` or `kite run ./hello.star`) rather than repository-relative directories (e.g., `./examples/core/hello.star`) to prevent logical disconnects.

## 3. Go Code Hygiene & CI Pre-Flight Checks
* **Mandatory Verification Suite**: Always run the complete Go code hygiene and pre-flight suite after making any code changes before completing a turn to guarantee remote CI passes:
  1. `go fix ./...` (apply standard AST modernizations such as `errors.AsType`, typed atomics, etc.)
  2. `go fmt ./...`
  3. `gofmt -l .` (verify zero formatting differences or trailing whitespace/newline issues)
  4. `go vet ./...`
  5. `go test -race ./...` (ensure data-race freedom and 100% test pass rate)
  6. `golangci-lint run ./...` (ensure 0 lint/shadowing/error-return warnings)
* **Verify Clean Working Tree**: Always inspect working copy status (`jj status` / `jj diff`) to confirm that all automatic modifications made by `go fix` and `go fmt` are captured in `@` so remote CI (`git diff --exit-code`) passes cleanly.
