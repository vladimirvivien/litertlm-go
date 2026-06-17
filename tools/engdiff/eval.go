package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Tier 2: a fixed eval set run greedy on both engines; the gate is the
// Go engine's accuracy within a small delta of the C++ engine's on the
// same model. Absolute accuracy is irrelevant (small models score low);
// a cross-engine gap localizes a templating, tokenization, or numeric
// defect that single-prompt byte diffs miss.

// evalItem is one eval-set entry. Kind selects the scorer:
//   - "number": the first number in the reply must equal Answer
//   - "contains": the reply must contain Answer (case-insensitive)
//   - "choice": the first standalone A–D letter must equal Answer
type evalItem struct {
	ID     string `json:"id"`
	Prompt string `json:"prompt"`
	Kind   string `json:"kind"`
	Answer string `json:"answer"`
}

func loadEvalSet(path string) ([]evalItem, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var items []evalItem
	if err := json.Unmarshal(data, &items); err != nil {
		return nil, fmt.Errorf("parse eval set: %w", err)
	}
	if len(items) == 0 {
		return nil, fmt.Errorf("eval set %s is empty", path)
	}
	return items, nil
}

var (
	numberRe = regexp.MustCompile(`-?\d+(?:\.\d+)?`)
	choiceRe = regexp.MustCompile(`\b([A-D])\b`)
)

// normNumber strips leading zeros so "042" matches "42".
func normNumber(s string) string {
	neg := strings.HasPrefix(s, "-")
	s = strings.TrimLeft(strings.TrimPrefix(s, "-"), "0")
	if s == "" {
		return "0"
	}
	if neg {
		return "-" + s
	}
	return s
}

// scoreReply reports whether reply answers the item.
func scoreReply(item evalItem, reply string) bool {
	r := strings.TrimSpace(reply)
	switch item.Kind {
	case "number":
		m := numberRe.FindString(r)
		if m == "" {
			return false
		}
		return normNumber(m) == normNumber(item.Answer)
	case "contains":
		return strings.Contains(strings.ToLower(r), strings.ToLower(item.Answer))
	case "choice":
		m := choiceRe.FindStringSubmatch(strings.ToUpper(r))
		return m != nil && m[1] == item.Answer
	default:
		return false
	}
}

// runEval executes the Tier 2 eval (and Tier 3 perf capture) for each
// model and reports per-engine accuracy. Returns false when any model's
// cross-engine delta exceeds maxDelta percentage points, a worker
// fails, or the perf baseline gate trips.
func runEval(models []string, setPath string, maxDelta float64, perfOut, perfBaseline, cppLib, goLib string) bool {
	items, err := loadEvalSet(setPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "engdiff:", err)
		return false
	}
	prompts := make([]string, len(items))
	for i, it := range items {
		prompts[i] = it.Prompt
	}

	baseline, err := loadPerfBaseline(perfBaseline)
	if err != nil {
		fmt.Fprintln(os.Stderr, "engdiff:", err)
		return false
	}

	ok := true
	for _, m := range models {
		name := filepath.Base(m)
		fmt.Printf("=== %s (%d items) ===\n", name, len(items))

		var scores [2]int
		var perfs [2]perfRecord
		engines := [2]string{"cpp", "go"}
		failed := false
		var mismatches []string
		var replies [2][]string

		for e, engine := range engines {
			r, perf, err := captureBatch(engine, m, prompts, cppLib, goLib)
			if err != nil {
				fmt.Printf("ERROR %s: %v\n", engine, err)
				failed = true
				break
			}
			replies[e] = r
			perfs[e] = perf
			for i, it := range items {
				if scoreReply(it, r[i]) {
					scores[e]++
				}
			}
		}
		if failed {
			ok = false
			continue
		}

		for i, it := range items {
			cppOK := scoreReply(it, replies[0][i])
			goOK := scoreReply(it, replies[1][i])
			if cppOK != goOK {
				mismatches = append(mismatches, fmt.Sprintf(
					"  %-12s cpp=%-5v go=%-5v cpp:%q go:%q",
					it.ID, cppOK, goOK, clip(replies[0][i], 40), clip(replies[1][i], 40)))
			}
		}

		n := float64(len(items))
		accCpp := 100 * float64(scores[0]) / n
		accGo := 100 * float64(scores[1]) / n
		d := accGo - accCpp
		verdict := "OK"
		if d < -maxDelta {
			verdict = "FAIL"
			ok = false
		}
		fmt.Printf("accuracy: cpp %.1f%% (%d/%d)  go %.1f%% (%d/%d)  delta %+.1fpp  [%s]\n",
			accCpp, scores[0], len(items), accGo, scores[1], len(items), d, verdict)
		for _, mm := range mismatches {
			fmt.Println(mm)
		}
		for e := range engines {
			p := perfs[e]
			fmt.Printf("perf %-3s: init %.1fs, %d reply tokens in %.1fs (%.1f tok/s end-to-end)\n",
				p.Engine, p.InitSecs, p.ReplyTokens, p.ReplySecs, p.TokPerSec)
			if !checkPerfBaseline(baseline, p) {
				ok = false
			}
			if perfOut != "" {
				if err := appendPerf(perfOut, p); err != nil {
					fmt.Fprintln(os.Stderr, "engdiff: perf-out:", err)
					ok = false
				}
			}
		}
		fmt.Println()
	}
	return ok
}

func appendPerf(path string, p perfRecord) error {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	return json.NewEncoder(f).Encode(p)
}

// loadPerfBaseline reads a perf JSONL keyed by model+engine; the last
// record per key wins.
func loadPerfBaseline(path string) (map[string]perfRecord, error) {
	if path == "" {
		return nil, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	out := map[string]perfRecord{}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var p perfRecord
		if err := json.Unmarshal([]byte(line), &p); err != nil {
			return nil, fmt.Errorf("perf baseline: %w", err)
		}
		out[p.Model+"/"+p.Engine] = p
	}
	return out, nil
}

// checkPerfBaseline flags a >10% regression in init time or end-to-end
// throughput against the recorded baseline. Missing keys pass.
func checkPerfBaseline(baseline map[string]perfRecord, p perfRecord) bool {
	if baseline == nil {
		return true
	}
	b, exists := baseline[p.Model+"/"+p.Engine]
	if !exists {
		return true
	}
	ok := true
	if b.InitSecs > 0 && p.InitSecs > b.InitSecs*1.10 {
		fmt.Printf("perf %-3s: FAIL init %.1fs vs baseline %.1fs (>10%%)\n", p.Engine, p.InitSecs, b.InitSecs)
		ok = false
	}
	if b.TokPerSec > 0 && p.TokPerSec < b.TokPerSec*0.90 {
		fmt.Printf("perf %-3s: FAIL %.1f tok/s vs baseline %.1f (>10%%)\n", p.Engine, p.TokPerSec, b.TokPerSec)
		ok = false
	}
	return ok
}
