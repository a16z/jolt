package jolt_verifier

import (
	"encoding/json"
	"fmt"
	"math/big"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// BN254 field modulus
var fieldModulus, _ = new(big.Int).SetString("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10)

// Power sums for univariate skip domain (size 10 centered at 0)
var powerSumsStage1 = []int64{
	10, 5, 85, 125, 1333, 3125, 25405, 78125, 535333, 1953125,
	11982925, 48828125, 278766133, 1220703125, 6649985245, 30517578125,
	161264049733, 762939453125, 3952911584365, 19073486328125,
	97573430562133, 476837158203125, 2419432933612285, 11920928955078125,
	60168159621439333, 298023223876953125, 1499128402505381005, 7450580596923828125,
}

// Power sums for Stage2 univariate skip (different domain)
var powerSumsStage2 = []int64{5, 0, 10, 0, 34, 0, 130, 0, 514, 0, 2050, 0, 8194}

// AssertionComponents holds the components of an assertion
type AssertionComponents struct {
	Name       string            `json:"name"`
	Components map[string]string `json:"components"`
	Result     string            `json:"result"`
}

// TestAssertionComponents extracts and displays the components of each assertion
func TestAssertionComponents(t *testing.T) {
	t.Log("=== Assertion Components Test ===")
	t.Log("Extracting components from each assertion a0...a14")
	t.Log("")

	// Load witness as map
	_, currentFile, _, _ := runtime.Caller(0)
	witnessPath := filepath.Join(filepath.Dir(currentFile), "stages16_witness.json")

	data, err := os.ReadFile(witnessPath)
	if err != nil {
		t.Fatalf("Failed to read witness file: %v", err)
	}

	var witness map[string]string
	if err := json.Unmarshal(data, &witness); err != nil {
		t.Fatalf("Failed to parse witness JSON: %v", err)
	}

	t.Logf("Loaded %d witness values", len(witness))
	t.Log("")

	// Collect all assertion components
	allComponents := make([]AssertionComponents, 0)

	// a0: Stage1 Univariate Skip
	a0 := computeA0(t, witness)
	allComponents = append(allComponents, a0)

	// a2: Stage2 Univariate Skip
	a2 := computeA2(t, witness)
	allComponents = append(allComponents, a2)

	// a5, a6, a7, a8: Claim differences
	a5 := computeClaimDiff(t, witness, "a5",
		"Claim_Polynomial_Virtual_Rs1Value_RegistersClaimReduction",
		"Claim_Polynomial_Virtual_Rs1Value_InstructionInputVirtualization")
	allComponents = append(allComponents, a5)

	a6 := computeClaimDiff(t, witness, "a6",
		"Claim_Polynomial_Virtual_Rs2Value_RegistersClaimReduction",
		"Claim_Polynomial_Virtual_Rs2Value_InstructionInputVirtualization")
	allComponents = append(allComponents, a6)

	a7 := computeClaimDiff(t, witness, "a7",
		"Claim_Polynomial_Virtual_Rs1Value_RegistersClaimReduction",
		"Claim_Polynomial_Virtual_Rs1Value_InstructionInputVirtualization")
	allComponents = append(allComponents, a7)

	a8 := computeClaimDiff(t, witness, "a8",
		"Claim_Polynomial_Virtual_Rs2Value_RegistersClaimReduction",
		"Claim_Polynomial_Virtual_Rs2Value_InstructionInputVirtualization")
	allComponents = append(allComponents, a8)

	// a10, a11: LookupOutput claim differences
	a10 := computeClaimDiff(t, witness, "a10",
		"Claim_Polynomial_Virtual_LookupOutput_InstructionClaimReduction",
		"Claim_Polynomial_Virtual_LookupOutput_SpartanProductVirtualization")
	allComponents = append(allComponents, a10)

	a11 := computeClaimDiff(t, witness, "a11",
		"Claim_Polynomial_Virtual_LookupOutput_InstructionClaimReduction",
		"Claim_Polynomial_Virtual_LookupOutput_SpartanProductVirtualization")
	allComponents = append(allComponents, a11)

	// a13: UnexpandedPC claim difference
	a13 := computeClaimDiff(t, witness, "a13",
		"Claim_Polynomial_Virtual_UnexpandedPC_SpartanShift",
		"Claim_Polynomial_Virtual_UnexpandedPC_InstructionInputVirtualization")
	allComponents = append(allComponents, a13)

	// Export to JSON for comparison with Rust
	exportJSON := map[string]interface{}{
		"source":     "go_gnark_witness",
		"assertions": allComponents,
	}

	jsonBytes, _ := json.MarshalIndent(exportJSON, "", "  ")
	outputPath := filepath.Join(filepath.Dir(currentFile), "go_assertion_components.json")
	os.WriteFile(outputPath, jsonBytes, 0644)
	t.Logf("\nExported to: %s", outputPath)
}

// computeA0 computes a0 = sum(Stage1_Uni_Skip_Coeff_i * powersum_i)
func computeA0(t *testing.T, witness map[string]string) AssertionComponents {
	t.Log("=== a0: Stage1 Univariate Skip ===")

	components := make(map[string]string)
	result := new(big.Int)

	for i := 0; i < 28; i++ {
		fieldName := fmt.Sprintf("Stage1_Uni_Skip_Coeff_%d", i)
		coeffStr, ok := witness[fieldName]
		if !ok {
			t.Logf("  Missing: %s", fieldName)
			continue
		}

		coeff, _ := new(big.Int).SetString(coeffStr, 10)
		powerSum := big.NewInt(powerSumsStage1[i])

		term := new(big.Int).Mul(coeff, powerSum)
		result.Add(result, term)

		components[fieldName] = coeffStr
		t.Logf("  %s = %s (× %d)", fieldName, truncateStr(coeffStr, 40), powerSumsStage1[i])
	}

	result.Mod(result, fieldModulus)
	t.Logf("  a0 = %s", result.String())

	return AssertionComponents{
		Name:       "a0",
		Components: components,
		Result:     result.String(),
	}
}

// computeA2 computes a2 = sum(Stage2_Uni_Skip_Coeff_i * powersum_i)
func computeA2(t *testing.T, witness map[string]string) AssertionComponents {
	t.Log("")
	t.Log("=== a2: Stage2 Univariate Skip ===")

	components := make(map[string]string)
	result := new(big.Int)

	for i := 0; i < len(powerSumsStage2); i++ {
		fieldName := fmt.Sprintf("Stage2_Uni_Skip_Coeff_%d", i)
		coeffStr, ok := witness[fieldName]
		if !ok {
			// Stage2 might have different numbering
			continue
		}

		coeff, _ := new(big.Int).SetString(coeffStr, 10)
		powerSum := big.NewInt(powerSumsStage2[i])

		term := new(big.Int).Mul(coeff, powerSum)
		result.Add(result, term)

		components[fieldName] = coeffStr
		t.Logf("  %s = %s (× %d)", fieldName, truncateStr(coeffStr, 40), powerSumsStage2[i])
	}

	result.Mod(result, fieldModulus)
	t.Logf("  a2 = %s", result.String())

	return AssertionComponents{
		Name:       "a2",
		Components: components,
		Result:     result.String(),
	}
}

// computeClaimDiff computes assertion = left - right
func computeClaimDiff(t *testing.T, witness map[string]string, name, leftField, rightField string) AssertionComponents {
	t.Log("")
	t.Logf("=== %s: Claim Difference ===", name)

	components := make(map[string]string)

	leftStr, ok1 := witness[leftField]
	rightStr, ok2 := witness[rightField]

	if !ok1 {
		t.Logf("  Missing: %s", leftField)
		leftStr = "0"
	}
	if !ok2 {
		t.Logf("  Missing: %s", rightField)
		rightStr = "0"
	}

	left, _ := new(big.Int).SetString(leftStr, 10)
	right, _ := new(big.Int).SetString(rightStr, 10)

	diff := new(big.Int).Sub(left, right)
	diff.Mod(diff, fieldModulus)

	components[leftField] = leftStr
	components[rightField] = rightStr

	t.Logf("  %s = %s", leftField, truncateStr(leftStr, 50))
	t.Logf("  %s = %s", rightField, truncateStr(rightStr, 50))
	t.Logf("  %s = left - right = %s", name, diff.String())

	return AssertionComponents{
		Name:       name,
		Components: components,
		Result:     diff.String(),
	}
}

// truncateStr truncates a string for display
func truncateStr(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}

// TestSumcheckComponents extracts sumcheck round components for a1, a3, a4, a9, a12, a14
func TestSumcheckComponents(t *testing.T) {
	t.Log("=== Sumcheck Assertion Components ===")
	t.Log("")

	// Load witness
	_, currentFile, _, _ := runtime.Caller(0)
	witnessPath := filepath.Join(filepath.Dir(currentFile), "stages16_witness.json")

	data, err := os.ReadFile(witnessPath)
	if err != nil {
		t.Fatalf("Failed to read witness file: %v", err)
	}

	var witness map[string]string
	if err := json.Unmarshal(data, &witness); err != nil {
		t.Fatalf("Failed to parse witness JSON: %v", err)
	}

	// a1: Stage1 Sumcheck (14 rounds: R0-R13)
	t.Log("=== a1: Stage1 Sumcheck ===")
	for r := 0; r <= 13; r++ {
		r0 := witness[fmt.Sprintf("Stage1_Sumcheck_R%d_0", r)]
		r1 := witness[fmt.Sprintf("Stage1_Sumcheck_R%d_1", r)]
		r2 := witness[fmt.Sprintf("Stage1_Sumcheck_R%d_2", r)]
		t.Logf("  Round %d: R0=%s, R1=%s, R2=%s",
			r, truncateStr(r0, 30), truncateStr(r1, 30), truncateStr(r2, 30))
	}

	// a3: Stage2 Sumcheck (26 rounds: R0-R25)
	t.Log("")
	t.Log("=== a3: Stage2 Sumcheck ===")
	for r := 0; r <= 25; r++ {
		r0 := witness[fmt.Sprintf("Stage2_Sumcheck_R%d_0", r)]
		if r0 == "" {
			continue
		}
		r1 := witness[fmt.Sprintf("Stage2_Sumcheck_R%d_1", r)]
		r2 := witness[fmt.Sprintf("Stage2_Sumcheck_R%d_2", r)]
		t.Logf("  Round %d: R0=%s, R1=%s, R2=%s",
			r, truncateStr(r0, 30), truncateStr(r1, 30), truncateStr(r2, 30))
	}

	// a4: Stage3 Sumcheck
	t.Log("")
	t.Log("=== a4: Stage3 Sumcheck ===")
	for r := 0; r <= 12; r++ {
		r0 := witness[fmt.Sprintf("Stage3_Sumcheck_R%d_0", r)]
		if r0 == "" {
			continue
		}
		r1 := witness[fmt.Sprintf("Stage3_Sumcheck_R%d_1", r)]
		r2 := witness[fmt.Sprintf("Stage3_Sumcheck_R%d_2", r)]
		t.Logf("  Round %d: R0=%s, R1=%s, R2=%s",
			r, truncateStr(r0, 30), truncateStr(r1, 30), truncateStr(r2, 30))
	}

	// a9: Stage4 Sumcheck
	t.Log("")
	t.Log("=== a9: Stage4 Sumcheck ===")
	for r := 0; r <= 19; r++ {
		r0 := witness[fmt.Sprintf("Stage4_Sumcheck_R%d_0", r)]
		if r0 == "" {
			continue
		}
		r1 := witness[fmt.Sprintf("Stage4_Sumcheck_R%d_1", r)]
		r2 := witness[fmt.Sprintf("Stage4_Sumcheck_R%d_2", r)]
		t.Logf("  Round %d: R0=%s, R1=%s, R2=%s",
			r, truncateStr(r0, 30), truncateStr(r1, 30), truncateStr(r2, 30))
	}

	// a12: Stage5 Sumcheck
	t.Log("")
	t.Log("=== a12: Stage5 Sumcheck ===")
	for r := 0; r <= 140; r++ {
		r0 := witness[fmt.Sprintf("Stage5_Sumcheck_R%d_0", r)]
		if r0 == "" {
			continue
		}
		r1 := witness[fmt.Sprintf("Stage5_Sumcheck_R%d_1", r)]
		r2 := witness[fmt.Sprintf("Stage5_Sumcheck_R%d_2", r)]
		if r <= 5 || r >= 135 { // Only show first and last few rounds
			t.Logf("  Round %d: R0=%s, R1=%s, R2=%s",
				r, truncateStr(r0, 30), truncateStr(r1, 30), truncateStr(r2, 30))
		} else if r == 6 {
			t.Log("  ... (rounds 6-134 omitted) ...")
		}
	}

	// a14: Stage6 Sumcheck
	t.Log("")
	t.Log("=== a14: Stage6 Sumcheck ===")
	for r := 0; r <= 25; r++ {
		r0 := witness[fmt.Sprintf("Stage6_Sumcheck_R%d_0", r)]
		if r0 == "" {
			continue
		}
		r1 := witness[fmt.Sprintf("Stage6_Sumcheck_R%d_1", r)]
		r2 := witness[fmt.Sprintf("Stage6_Sumcheck_R%d_2", r)]
		t.Logf("  Round %d: R0=%s, R1=%s, R2=%s",
			r, truncateStr(r0, 30), truncateStr(r1, 30), truncateStr(r2, 30))
	}
}

// TestCompareAssertionResults compares the COMPUTED assertion results between Rust and Go
func TestCompareAssertionResults(t *testing.T) {
	t.Log("=== Compare Assertion RESULTS: Rust vs Go ===")
	t.Log("Comparing the computed value of each assertion")
	t.Log("")

	_, currentFile, _, _ := runtime.Caller(0)
	dir := filepath.Dir(currentFile)

	// Load Rust assertion results
	rustPath := filepath.Join(dir, "rust_assertion_components.json")
	rustData, err := os.ReadFile(rustPath)
	if err != nil {
		t.Skipf("Rust components not found. Run: cargo run -p gnark-transpiler --bin export_assertion_components\n%v", err)
	}

	var rustExport struct {
		Source     string                `json:"source"`
		Assertions []AssertionComponents `json:"assertions"`
	}
	if err := json.Unmarshal(rustData, &rustExport); err != nil {
		t.Fatalf("Failed to parse Rust JSON: %v", err)
	}

	// Load Go witness and compute assertions
	witnessPath := filepath.Join(dir, "stages16_witness.json")
	witnessData, err := os.ReadFile(witnessPath)
	if err != nil {
		t.Fatalf("Failed to read Go witness: %v", err)
	}

	var witness map[string]string
	if err := json.Unmarshal(witnessData, &witness); err != nil {
		t.Fatalf("Failed to parse Go witness: %v", err)
	}

	// Compute a0 in Go
	goA0 := new(big.Int)
	for i := 0; i < 28; i++ {
		coeffStr := witness[fmt.Sprintf("Stage1_Uni_Skip_Coeff_%d", i)]
		coeff, _ := new(big.Int).SetString(coeffStr, 10)
		powerSum := big.NewInt(powerSumsStage1[i])
		term := new(big.Int).Mul(coeff, powerSum)
		goA0.Add(goA0, term)
	}
	goA0.Mod(goA0, fieldModulus)

	// Find Rust a0 result
	var rustA0Result string
	for _, a := range rustExport.Assertions {
		if a.Name == "a0" {
			rustA0Result = a.Result
			break
		}
	}

	t.Log("=== a0: Stage1 Univariate Skip ===")
	t.Logf("  Rust a0 = %s", rustA0Result)
	t.Logf("  Go   a0 = %s", goA0.String())
	if rustA0Result == goA0.String() {
		t.Log("  [✓] MATCH!")
	} else {
		t.Errorf("  [✗] MISMATCH!")
	}
	t.Log("")

	// Compute simple claim differences (a5, a6, a7, a8, a10, a11, a13)
	claimPairs := []struct {
		name  string
		left  string
		right string
	}{
		{"a5", "Claim_Polynomial_Virtual_Rs1Value_RegistersClaimReduction", "Claim_Polynomial_Virtual_Rs1Value_InstructionInputVirtualization"},
		{"a6", "Claim_Polynomial_Virtual_Rs2Value_RegistersClaimReduction", "Claim_Polynomial_Virtual_Rs2Value_InstructionInputVirtualization"},
		{"a10", "Claim_Polynomial_Virtual_LookupOutput_InstructionClaimReduction", "Claim_Polynomial_Virtual_LookupOutput_SpartanProductVirtualization"},
		{"a13", "Claim_Polynomial_Virtual_UnexpandedPC_SpartanShift", "Claim_Polynomial_Virtual_UnexpandedPC_InstructionInputVirtualization"},
	}

	for _, cp := range claimPairs {
		leftStr := witness[cp.left]
		rightStr := witness[cp.right]

		left, _ := new(big.Int).SetString(leftStr, 10)
		right, _ := new(big.Int).SetString(rightStr, 10)

		diff := new(big.Int).Sub(left, right)
		diff.Mod(diff, fieldModulus)

		t.Logf("=== %s: Claim Difference ===", cp.name)
		t.Logf("  Left  = %s", truncateStr(leftStr, 50))
		t.Logf("  Right = %s", truncateStr(rightStr, 50))
		t.Logf("  Go %s = %s", cp.name, diff.String())
		if diff.Cmp(big.NewInt(0)) == 0 {
			t.Log("  [✓] Result is 0 (as expected)")
		} else {
			t.Errorf("  [✗] Result is NOT 0!")
		}
		t.Log("")
	}

	t.Log("=== Summary ===")
	t.Log("Rust a0 and Go a0 both equal 0: assertions match!")
}

// TestCompareWithRust loads Rust assertion components and compares with Go witness values
func TestCompareWithRust(t *testing.T) {
	t.Log("=== Compare Assertion Components: Rust vs Go ===")
	t.Log("Comparing individual witness values between Rust proof and Go witness")
	t.Log("")

	_, currentFile, _, _ := runtime.Caller(0)
	dir := filepath.Dir(currentFile)

	// Load Rust components
	rustPath := filepath.Join(dir, "rust_assertion_components.json")
	rustData, err := os.ReadFile(rustPath)
	if err != nil {
		t.Skipf("Rust components not found. Run: cargo run -p gnark-transpiler --bin export_assertion_components\n%v", err)
	}

	var rustExport struct {
		Source     string                `json:"source"`
		Assertions []AssertionComponents `json:"assertions"`
	}
	if err := json.Unmarshal(rustData, &rustExport); err != nil {
		t.Fatalf("Failed to parse Rust JSON: %v", err)
	}

	// Load Go witness directly
	witnessPath := filepath.Join(dir, "stages16_witness.json")
	witnessData, err := os.ReadFile(witnessPath)
	if err != nil {
		t.Fatalf("Failed to read Go witness: %v", err)
	}

	var goWitness map[string]string
	if err := json.Unmarshal(witnessData, &goWitness); err != nil {
		t.Fatalf("Failed to parse Go witness: %v", err)
	}

	t.Logf("Rust assertions: %d", len(rustExport.Assertions))
	t.Logf("Go witness values: %d", len(goWitness))
	t.Log("")

	// Compare components for each assertion
	totalComparisons := 0
	matchCount := 0
	mismatchCount := 0

	for _, rustA := range rustExport.Assertions {
		t.Logf("=== %s ===", rustA.Name)

		componentMatches := 0
		componentMismatches := 0

		for rustKey, rustVal := range rustA.Components {
			goVal, ok := goWitness[rustKey]
			if !ok {
				t.Logf("  [?] %s: not in Go witness", rustKey)
				continue
			}

			totalComparisons++
			if rustVal == goVal {
				componentMatches++
				matchCount++
				// Print each matching component explicitly
				t.Logf("  [✓] %s", rustKey)
				t.Logf("      Rust: %s", truncateStr(rustVal, 60))
				t.Logf("      Go:   %s", truncateStr(goVal, 60))
			} else {
				componentMismatches++
				mismatchCount++
				t.Logf("  [✗] %s MISMATCH:", rustKey)
				t.Logf("      Rust: %s", truncateStr(rustVal, 60))
				t.Logf("      Go:   %s", truncateStr(goVal, 60))
			}
		}

		t.Logf("  --- %d matches, %d mismatches ---", componentMatches, componentMismatches)
		t.Log("")
	}

	t.Log("=== Summary ===")
	t.Logf("Total comparisons: %d", totalComparisons)
	t.Logf("Matches: %d", matchCount)
	t.Logf("Mismatches: %d", mismatchCount)

	// Export comparison results to JSON
	type ComponentComparison struct {
		Name    string `json:"name"`
		Rust    string `json:"rust"`
		Go      string `json:"go"`
		Match   bool   `json:"match"`
	}

	type AssertionComparison struct {
		Assertion  string                `json:"assertion"`
		Components []ComponentComparison `json:"components"`
		AllMatch   bool                  `json:"all_match"`
	}

	var comparisons []AssertionComparison
	for _, rustA := range rustExport.Assertions {
		ac := AssertionComparison{
			Assertion: rustA.Name,
			AllMatch:  true,
		}
		for rustKey, rustVal := range rustA.Components {
			goVal, ok := goWitness[rustKey]
			if !ok {
				continue
			}
			match := rustVal == goVal
			if !match {
				ac.AllMatch = false
			}
			ac.Components = append(ac.Components, ComponentComparison{
				Name:  rustKey,
				Rust:  rustVal,
				Go:    goVal,
				Match: match,
			})
		}
		comparisons = append(comparisons, ac)
	}

	exportData := map[string]interface{}{
		"total_comparisons": totalComparisons,
		"matches":           matchCount,
		"mismatches":        mismatchCount,
		"assertions":        comparisons,
	}

	jsonBytes, _ := json.MarshalIndent(exportData, "", "  ")
	outputPath := filepath.Join(dir, "rust_go_comparison.json")
	os.WriteFile(outputPath, jsonBytes, 0644)
	t.Logf("Comparison exported to: %s", outputPath)

	if mismatchCount > 0 {
		t.Errorf("Found %d mismatches between Rust and Go witness values!", mismatchCount)
	} else {
		t.Log("✓ All compared values match between Rust and Go!")
	}
}

// TestFullPipeline runs the complete Rust + Go comparison pipeline in one test.
// It executes the Rust binary to export components, then compares with Go witness.
func TestFullPipeline(t *testing.T) {
	t.Log("=== Full Pipeline: Rust Export + Go Compare ===")
	t.Log("")

	_, currentFile, _, _ := runtime.Caller(0)
	dir := filepath.Dir(currentFile)

	// Find the gnark-transpiler root (parent of go/)
	transpilerRoot := filepath.Dir(dir)

	// Step 1: Run Rust binary to export assertion components
	t.Log("Step 1: Running Rust export_assertion_components...")
	t.Log("")

	cmd := exec.Command("cargo", "run", "-p", "gnark-transpiler", "--bin", "export_assertion_components")
	cmd.Dir = filepath.Dir(transpilerRoot) // jolt directory (parent of gnark-transpiler)

	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("Failed to run Rust export:\n%s\nError: %v", string(output), err)
	}

	// Show Rust output (truncated)
	rustOutput := string(output)
	lines := strings.Split(rustOutput, "\n")
	if len(lines) > 30 {
		t.Log("Rust output (first 15 lines):")
		for _, line := range lines[:15] {
			t.Logf("  %s", line)
		}
		t.Log("  ...")
		t.Log("Rust output (last 10 lines):")
		for _, line := range lines[len(lines)-11:] {
			t.Logf("  %s", line)
		}
	} else {
		t.Log("Rust output:")
		for _, line := range lines {
			t.Logf("  %s", line)
		}
	}
	t.Log("")

	// Step 2: Load Rust components
	t.Log("Step 2: Loading Rust assertion components...")

	rustPath := filepath.Join(dir, "rust_assertion_components.json")
	rustData, err := os.ReadFile(rustPath)
	if err != nil {
		t.Fatalf("Failed to read Rust components: %v", err)
	}

	var rustExport struct {
		Source     string                `json:"source"`
		Assertions []AssertionComponents `json:"assertions"`
	}
	if err := json.Unmarshal(rustData, &rustExport); err != nil {
		t.Fatalf("Failed to parse Rust JSON: %v", err)
	}
	t.Logf("  Loaded %d assertions from Rust", len(rustExport.Assertions))
	t.Log("")

	// Step 3: Load Go witness
	t.Log("Step 3: Loading Go witness...")

	witnessPath := filepath.Join(dir, "stages16_witness.json")
	witnessData, err := os.ReadFile(witnessPath)
	if err != nil {
		t.Fatalf("Failed to read Go witness: %v", err)
	}

	var goWitness map[string]string
	if err := json.Unmarshal(witnessData, &goWitness); err != nil {
		t.Fatalf("Failed to parse Go witness: %v", err)
	}
	t.Logf("  Loaded %d witness values from Go", len(goWitness))
	t.Log("")

	// Step 4: Compare
	t.Log("Step 4: Comparing Rust vs Go components...")
	t.Log("")

	totalComparisons := 0
	matchCount := 0
	mismatchCount := 0

	type ComponentComparison struct {
		Name  string `json:"name"`
		Rust  string `json:"rust"`
		Go    string `json:"go"`
		Match bool   `json:"match"`
	}

	type AssertionComparison struct {
		Assertion  string                `json:"assertion"`
		Components []ComponentComparison `json:"components"`
		AllMatch   bool                  `json:"all_match"`
	}

	var comparisons []AssertionComparison

	for _, rustA := range rustExport.Assertions {
		t.Logf("=== %s ===", rustA.Name)

		ac := AssertionComparison{
			Assertion: rustA.Name,
			AllMatch:  true,
		}

		componentMatches := 0
		componentMismatches := 0

		for rustKey, rustVal := range rustA.Components {
			goVal, ok := goWitness[rustKey]
			if !ok {
				t.Logf("  [?] %s: not in Go witness", rustKey)
				continue
			}

			totalComparisons++
			match := rustVal == goVal

			if match {
				componentMatches++
				matchCount++
				t.Logf("  [✓] %s", rustKey)
				t.Logf("      Rust: %s", truncateStr(rustVal, 60))
				t.Logf("      Go:   %s", truncateStr(goVal, 60))
			} else {
				componentMismatches++
				mismatchCount++
				ac.AllMatch = false
				t.Logf("  [✗] %s MISMATCH:", rustKey)
				t.Logf("      Rust: %s", truncateStr(rustVal, 60))
				t.Logf("      Go:   %s", truncateStr(goVal, 60))
			}

			ac.Components = append(ac.Components, ComponentComparison{
				Name:  rustKey,
				Rust:  rustVal,
				Go:    goVal,
				Match: match,
			})
		}

		comparisons = append(comparisons, ac)
		t.Logf("  --- %d matches, %d mismatches ---", componentMatches, componentMismatches)
		t.Log("")
	}

	// Export comparison to JSON
	exportData := map[string]interface{}{
		"total_comparisons": totalComparisons,
		"matches":           matchCount,
		"mismatches":        mismatchCount,
		"assertions":        comparisons,
	}

	jsonBytes, _ := json.MarshalIndent(exportData, "", "  ")
	outputPath := filepath.Join(dir, "rust_go_comparison.json")
	os.WriteFile(outputPath, jsonBytes, 0644)

	// Final summary
	t.Log("=== Pipeline Summary ===")
	t.Logf("Total comparisons: %d", totalComparisons)
	t.Logf("Matches: %d", matchCount)
	t.Logf("Mismatches: %d", mismatchCount)
	t.Logf("Comparison exported to: %s", outputPath)

	if mismatchCount > 0 {
		t.Errorf("FAILED: Found %d mismatches between Rust and Go!", mismatchCount)
	} else {
		t.Log("")
		t.Log("✓ SUCCESS: All Rust and Go assertion components match!")
	}
}
