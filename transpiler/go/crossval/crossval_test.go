package crossval

import (
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/test"
)

// loadAssignment loads witness data into the crossval circuit struct.
func loadAssignment(witnessPath string) (*JoltStagesCircuit, error) {
	data, err := os.ReadFile(witnessPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read witness file: %w", err)
	}

	var witnessMap map[string]string
	if err := json.Unmarshal(data, &witnessMap); err != nil {
		return nil, fmt.Errorf("failed to parse witness JSON: %w", err)
	}

	assignment := &JoltStagesCircuit{}
	v := reflect.ValueOf(assignment).Elem()
	t := v.Type()

	loaded := 0
	for i := 0; i < v.NumField(); i++ {
		fieldName := t.Field(i).Name
		field := v.Field(i)
		if value, ok := witnessMap[fieldName]; ok {
			if field.IsValid() && field.CanSet() {
				val, ok := new(big.Int).SetString(value, 10)
				if !ok {
					return nil, fmt.Errorf("invalid value for %s: %s", fieldName, value)
				}
				field.Set(reflect.ValueOf(val))
				loaded++
			}
		}
	}

	if loaded == 0 {
		return nil, fmt.Errorf("no witness values loaded")
	}
	return assignment, nil
}

// AssertionValue holds LHS and RHS for one assertion.
type AssertionValue struct {
	Name string `json:"name"`
	LHS  string `json:"lhs"`
	RHS  string `json:"rhs"`
}

// TestCrossValidation runs the debug circuit and captures api.Println output.
// The Println output goes through gnark's test engine and appears in test output.
// We parse it from the (test.engine) lines and write to JSON.
func TestCrossValidation(t *testing.T) {
	// Redirect stdout to capture api.Println output from test engine
	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	// Load witness
	_, currentFile, _, _ := runtime.Caller(0)
	witnessPath := filepath.Join(filepath.Dir(currentFile), "..", "stages_witness.json")
	assignment, err := loadAssignment(witnessPath)
	if err != nil {
		w.Close()
		os.Stdout = oldStdout
		t.Fatalf("Failed to load witness: %v", err)
	}

	// Solve circuit (triggers api.Println calls)
	var circuit JoltStagesCircuit
	err = test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())

	// Restore stdout and read captured output
	w.Close()
	os.Stdout = oldStdout

	capturedBytes, _ := io.ReadAll(r)
	capturedOutput := string(capturedBytes)

	if err != nil {
		t.Fatalf("Circuit not satisfied: %v\nCaptured output:\n%s", err, capturedOutput)
	}

	// Parse api.Println output from captured stdout
	// Format: (test.engine) circuit.go:NNNN KEY VALUE
	values := make(map[string]string)
	lines := strings.Split(capturedOutput, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.Contains(line, "test.engine") {
			continue
		}
		// Extract key-value pairs for assertion LHS/RHS
		for i := 0; i < 20; i++ {
			for _, suffix := range []string{"_lhs", "_rhs", "_total"} {
				key := fmt.Sprintf("a%d%s", i, suffix)
				idx := strings.Index(line, key)
				if idx < 0 {
					continue
				}
				rest := strings.TrimSpace(line[idx+len(key):])
				parts := strings.Fields(rest)
				if len(parts) > 0 {
					values[key] = parts[0]
				}
			}
		}
	}

	// Build assertion list
	assertions := make([]AssertionValue, 20)
	for i := 0; i < 20; i++ {
		lhs := values[fmt.Sprintf("a%d_lhs", i)]
		rhs := values[fmt.Sprintf("a%d_rhs", i)]

		// a0 is special: sum_zero, LHS = total, RHS = 0
		if i == 0 {
			if total, ok := values["a0_total"]; ok {
				lhs = total
				rhs = "0"
			}
		}

		assertions[i] = AssertionValue{
			Name: fmt.Sprintf("a%d", i),
			LHS:  lhs,
			RHS:  rhs,
		}
		t.Logf("a%d: LHS=%s RHS=%s", i, truncate(lhs, 50), truncate(rhs, 50))
	}

	// Verify we got all values
	missing := 0
	for i, a := range assertions {
		if a.LHS == "" {
			t.Errorf("Missing LHS for a%d", i)
			missing++
		}
		if a.RHS == "" && i != 0 {
			t.Errorf("Missing RHS for a%d", i)
			missing++
		}
	}

	// Write to JSON
	jsonBytes, err := json.MarshalIndent(assertions, "", "  ")
	if err != nil {
		t.Fatalf("Failed to marshal JSON: %v", err)
	}
	outPath := filepath.Join(filepath.Dir(currentFile), "..", "go_crossval_values.json")
	if err := os.WriteFile(outPath, jsonBytes, 0644); err != nil {
		t.Fatalf("Failed to write JSON: %v", err)
	}
	t.Logf("Wrote %d assertions to %s (missing: %d)", len(assertions), outPath, missing)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-3] + "..."
}
