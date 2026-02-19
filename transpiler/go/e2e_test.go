package jolt_verifier

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// getWorkspaceRoot returns the Cargo workspace root (jolt/)
func getWorkspaceRoot() string {
	_, currentFile, _, _ := runtime.Caller(0)
	return filepath.Dir(filepath.Dir(filepath.Dir(currentFile)))
}

func runCommand(t *testing.T, name string, dir string, bin string, args ...string) time.Duration {
	t.Helper()
	cmd := exec.Command(bin, args...)
	cmd.Dir = dir
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr

	t.Logf("Running: %s %v", bin, args)
	start := time.Now()
	if err := cmd.Run(); err != nil {
		t.Fatalf("%s failed: %v", name, err)
	}
	elapsed := time.Since(start)
	t.Logf("%s completed [%v]", name, elapsed)
	return elapsed
}

// TestEndToEndPipeline runs the complete pipeline and reports all timings.
//
// Usage: go test -run TestEndToEndPipeline -v -timeout 30m
func TestEndToEndPipeline(t *testing.T) {
	t.Log("=== End-to-End Pipeline ===")
	root := getWorkspaceRoot()
	_, thisFile, _, _ := runtime.Caller(0)
	goDir := filepath.Dir(thisFile)

	// Step 0: Build Rust binaries (not timed)
	t.Log("--- Step 0: Building Rust binaries ---")
	runCommand(t, "build-fibonacci", root,
		"cargo", "build", "-p", "fibonacci", "--release",
		"--features", "transcript-poseidon",
	)
	runCommand(t, "build-transpiler", root,
		"cargo", "build", "-p", "transpiler", "--bin", "transpiler",
	)
	t.Log("Rust binaries ready")

	totalStart := time.Now()

	// Step 1: Fibonacci proof (binary only, no compilation)
	t.Log("--- Step 1: Fibonacci Proof ---")
	fibBin := filepath.Join(root, "target", "release", "fibonacci")
	fibTime := runCommand(t, "fibonacci", root,
		fibBin, "--save", "50",
	)

	// Step 2: Transpile (binary only, no compilation)
	t.Log("--- Step 2: Transpile ---")
	transpilerBin := filepath.Join(root, "target", "debug", "transpiler")
	transpileTime := runCommand(t, "transpiler", root,
		transpilerBin, "--target", "gnark",
	)

	// Step 3: Groth16 (subprocess to pick up regenerated circuit)
	t.Log("--- Step 3: Groth16 ---")
	groth16Time := runCommand(t, "groth16", goDir,
		"go", "test", "-run", "TestStagesCircuitProveVerify",
		"-v", "-timeout", "25m", "-count=1",
	)

	// Read detailed results written by TestStagesCircuitProveVerify
	resultsPath := filepath.Join(goDir, "groth16_results.json")
	data, err := os.ReadFile(resultsPath)
	if err != nil {
		t.Fatalf("Failed to read groth16_results.json: %v", err)
	}
	var results map[string]float64
	if err := json.Unmarshal(data, &results); err != nil {
		t.Fatalf("Failed to parse groth16_results.json: %v", err)
	}

	totalTime := time.Since(totalStart)
	t.Log("")
	t.Log("========================================")
	t.Log("=== End-to-End Summary ===")
	t.Log("========================================")
	t.Logf("Fibonacci proof (Rust):  %v", fibTime)
	t.Logf("Transpile (Rust):        %v", transpileTime)
	t.Logf("Circuit compile (Go):    %v", time.Duration(results["compile_ms"])*time.Millisecond)
	t.Logf("Groth16 setup (Go):      %v", time.Duration(results["setup_ms"])*time.Millisecond)
	t.Logf("Groth16 prove (Go):      %v", time.Duration(results["prove_ms"])*time.Millisecond)
	t.Logf("Groth16 verify (Go):     %v", time.Duration(results["verify_ms"])*time.Millisecond)
	t.Log("----------------------------------------")
	t.Logf("TOTAL pipeline:          %v", totalTime)
	t.Logf("Groth16 total:           %v", groth16Time)
	t.Logf("Constraints:             %.0f", results["constraints"])
	t.Logf("Proof size:              %.0f bytes", results["proof_bytes"])
	t.Log("========================================")
}

// loadWitnessMap is a helper to load raw witness JSON
func loadWitnessMap(path string) (map[string]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read: %w", err)
	}
	var m map[string]string
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("failed to parse: %w", err)
	}
	return m, nil
}
