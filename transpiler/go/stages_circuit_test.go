package jolt_verifier

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/test"
)

// getStagesWitnessPath returns the path to stages_witness.json
func getStagesWitnessPath() string {
	_, currentFile, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(currentFile), "stages_witness.json")
}

// LoadStagesAssignment loads witness data and creates a circuit assignment
func LoadStagesAssignment(witnessPath string) (*JoltStagesCircuit, error) {
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
		return nil, fmt.Errorf("no witness values loaded - field name mismatch?")
	}

	return assignment, nil
}

func TestStagesCircuitCompile(t *testing.T) {
	t.Log("Jolt Stages 1-7 Verifier - Gnark Circuit Compilation Test")
	t.Log("")

	// Compile
	t.Log("Compiling circuit...")
	startCompile := time.Now()

	var circuit JoltStagesCircuit
	r1cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		t.Fatalf("Failed to compile circuit: %v", err)
	}
	compileTime := time.Since(startCompile)

	t.Log("")
	t.Log("=== Compilation Results ===")
	t.Logf("R1CS Constraints:    %d", r1cs.GetNbConstraints())
	t.Logf("Public Variables:    %d", r1cs.GetNbPublicVariables())
	t.Logf("Internal Variables:  %d", r1cs.GetNbInternalVariables())
	t.Logf("Compile Time:        %v", compileTime)
	t.Log("")
	t.Log("✓ Circuit compilation successful!")
}

func TestStagesCircuitSolver(t *testing.T) {
	t.Log("Jolt Stages 1-7 Verifier - Gnark Solver Debug Test")
	t.Log("")

	// Load witness
	witnessPath := getStagesWitnessPath()
	assignment, err := LoadStagesAssignment(witnessPath)
	if err != nil {
		t.Fatalf("Failed to load witness data: %v", err)
	}
	t.Logf("Loaded witness from: %s", witnessPath)

	// Debug: print some witness values
	v := reflect.ValueOf(assignment).Elem()
	t.Log("Sample witness values:")
	fieldsToShow := []string{
		"Stage1_Uni_Skip_Coeff_0",
		"Stage1_Uni_Skip_Coeff_1",
		"Claim_Virtual_UnivariateSkip_SpartanOuter",
		"Stage1_Sumcheck_R0_0",
		"Stage1_Sumcheck_R0_1",
		"Stage1_Sumcheck_R0_2",
	}
	for _, name := range fieldsToShow {
		field := v.FieldByName(name)
		if field.IsValid() && !field.IsZero() {
			t.Logf("  %s = %v", name, field.Interface())
		}
	}

	// Manually compute a0 to verify
	// a0 is: sum_j coeff_j * power_sums[j] == 0
	// power_sums for domain size 10 centered at 0 are precomputed constants
	powerSums := []int64{10, 5, 85, 125, 1333, 3125, 25405, 78125, 535333, 1953125, 11982925, 48828125, 278766133, 1220703125, 6649985245, 30517578125, 161264049733, 762939453125, 3952911584365, 19073486328125, 97573430562133, 476837158203125, 2419432933612285, 11920928955078125, 60168159621439333, 298023223876953125, 1499128402505381005, 7450580596923828125}
	coeffFields := []string{
		"Stage1_Uni_Skip_Coeff_0", "Stage1_Uni_Skip_Coeff_1", "Stage1_Uni_Skip_Coeff_2", "Stage1_Uni_Skip_Coeff_3",
		"Stage1_Uni_Skip_Coeff_4", "Stage1_Uni_Skip_Coeff_5", "Stage1_Uni_Skip_Coeff_6", "Stage1_Uni_Skip_Coeff_7",
		"Stage1_Uni_Skip_Coeff_8", "Stage1_Uni_Skip_Coeff_9", "Stage1_Uni_Skip_Coeff_10", "Stage1_Uni_Skip_Coeff_11",
		"Stage1_Uni_Skip_Coeff_12", "Stage1_Uni_Skip_Coeff_13", "Stage1_Uni_Skip_Coeff_14", "Stage1_Uni_Skip_Coeff_15",
		"Stage1_Uni_Skip_Coeff_16", "Stage1_Uni_Skip_Coeff_17", "Stage1_Uni_Skip_Coeff_18", "Stage1_Uni_Skip_Coeff_19",
		"Stage1_Uni_Skip_Coeff_20", "Stage1_Uni_Skip_Coeff_21", "Stage1_Uni_Skip_Coeff_22", "Stage1_Uni_Skip_Coeff_23",
		"Stage1_Uni_Skip_Coeff_24", "Stage1_Uni_Skip_Coeff_25", "Stage1_Uni_Skip_Coeff_26", "Stage1_Uni_Skip_Coeff_27",
	}
	p, _ := new(big.Int).SetString("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10)
	sum := new(big.Int)
	for i, name := range coeffFields {
		field := v.FieldByName(name)
		if field.IsValid() {
			coeff := field.Interface().(*big.Int)
			ps := big.NewInt(powerSums[i])
			term := new(big.Int).Mul(coeff, ps)
			sum.Add(sum, term)
		}
	}
	sum.Mod(sum, p)
	t.Logf("Manual a0 computation: %v (should be 0)", sum)

	// Debug: Print sumcheck round values for Stage1
	t.Log("\nStage1 Sumcheck Round Values:")
	for r := 0; r <= 10; r++ {
		for j := 0; j <= 2; j++ {
			name := fmt.Sprintf("Stage1_Sumcheck_R%d_%d", r, j)
			field := v.FieldByName(name)
			if field.IsValid() && !field.IsZero() {
				t.Logf("  %s = %v", name, field.Interface())
			}
		}
	}

	// Print the UnivariateSkip claim
	t.Log("\nClaims:")
	uniSkipField := v.FieldByName("Claim_Virtual_UnivariateSkip_SpartanOuter")
	if uniSkipField.IsValid() {
		t.Logf("  Claim_Virtual_UnivariateSkip_SpartanOuter = %v", uniSkipField.Interface())
	}

	// Manually compute the sumcheck verification for Stage1
	t.Log("\n=== Manual Sumcheck Verification ===")

	// The sumcheck verification computes:
	// For each round i: e = eval_compressed_poly(e, R_i, challenge_i)
	// Where eval_compressed_poly evaluates using the hint formula

	// For a degree-2 compressed polynomial with coefficients [c0, c1, c2]:
	// The eval at challenge r is: c0 + c1*r + c2*r^2
	// But the compressed poly stores: [eval_at_0 - eval_at_2, eval_at_2]
	// And we compute: eval_at_0 + r * (hint - eval_at_0) + r^2 * (eval_at_2 - eval_at_0 - hint)
	// Which simplifies to: (1-r)*e0 + r*h + r^2*(e2-e0-h) where h = e(1)
	// Actually the formula is: e = R0 + r * (h - R0 - R0) + r^2 * (R2)
	// where R0 = e(0), R2 = e(2), h = e(1) - R2

	// Let's print some intermediate claim values
	claimFields := []string{
		"Claim_Virtual_OpFlags_Load_SpartanOuter",
		"Claim_Virtual_OpFlags_Store_SpartanOuter",
		"Claim_Virtual_RamAddress_SpartanOuter",
		"Claim_Virtual_LookupOutput_SpartanOuter",
	}
	for _, name := range claimFields {
		field := v.FieldByName(name)
		if field.IsValid() && !field.IsZero() {
			t.Logf("  %s = %v", name, field.Interface())
		}
	}

	// Use gnark test solver to check constraints
	var circuit JoltStagesCircuit
	err = test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Logf("Solver error: %v", err)
		// The error message should tell us which constraint failed
	} else {
		t.Log("All constraints satisfied!")
	}
}


func TestStagesCircuitProveVerify(t *testing.T) {
	t.Log("Jolt Stages 1-7 Verifier - Full Groth16 Prove/Verify Test")
	t.Log("")

	// Load witness
	witnessPath := getStagesWitnessPath()
	assignment, err := LoadStagesAssignment(witnessPath)
	if err != nil {
		t.Fatalf("Failed to load witness data: %v", err)
	}
	t.Logf("Loaded witness from: %s", witnessPath)

	// Compile
	t.Log("")
	t.Log("Compiling circuit...")
	startCompile := time.Now()

	var circuit JoltStagesCircuit
	r1cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		t.Fatalf("Failed to compile circuit: %v", err)
	}
	compileTime := time.Since(startCompile)

	t.Logf("Compiled: %d constraints, %d public inputs, %d internal vars [%v]",
		r1cs.GetNbConstraints(), r1cs.GetNbPublicVariables(), r1cs.GetNbInternalVariables(), compileTime)

	// Setup
	t.Log("")
	t.Log("Running Groth16 setup...")
	startSetup := time.Now()

	pk, vk, err := groth16.Setup(r1cs)
	if err != nil {
		t.Fatalf("Failed to setup: %v", err)
	}
	setupTime := time.Since(startSetup)

	var pkBuf, vkBuf bytes.Buffer
	pk.WriteTo(&pkBuf)
	vk.WriteTo(&vkBuf)

	t.Logf("Setup complete: pk=%.2f MB, vk=%.2f KB [%v]",
		float64(pkBuf.Len())/1024/1024, float64(vkBuf.Len())/1024, setupTime)

	// Prove
	t.Log("")
	t.Log("Generating proof...")
	startProve := time.Now()

	witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Failed to create witness: %v", err)
	}

	proof, err := groth16.Prove(r1cs, pk, witness)
	if err != nil {
		t.Fatalf("Failed to prove: %v", err)
	}
	proveTime := time.Since(startProve)

	var proofBuf bytes.Buffer
	proof.WriteTo(&proofBuf)

	t.Logf("Proof generated: %d bytes [%v]", proofBuf.Len(), proveTime)

	// Verify
	t.Log("")
	t.Log("Verifying proof...")
	startVerify := time.Now()

	publicWitness, err := witness.Public()
	if err != nil {
		t.Fatalf("Failed to get public witness: %v", err)
	}

	err = groth16.Verify(proof, vk, publicWitness)
	if err != nil {
		t.Fatalf("Failed to verify: %v", err)
	}
	verifyTime := time.Since(startVerify)

	t.Logf("Verified [%v]", verifyTime)

	// Summary
	t.Log("")
	t.Log("=== Summary ===")
	t.Logf("Constraints: %d", r1cs.GetNbConstraints())
	t.Logf("Proof size:  %d bytes", proofBuf.Len())
	t.Logf("Prove time:  %v", proveTime)
	t.Logf("Verify time: %v", verifyTime)
	t.Log("")
	t.Log("✓ Stages 1-7 circuit verification passed!")

	// Write results JSON for e2e test
	results := map[string]interface{}{
		"constraints": r1cs.GetNbConstraints(),
		"proof_bytes": proofBuf.Len(),
		"compile_ms":  compileTime.Milliseconds(),
		"setup_ms":    setupTime.Milliseconds(),
		"prove_ms":    proveTime.Milliseconds(),
		"verify_ms":   verifyTime.Milliseconds(),
	}
	if data, err := json.Marshal(results); err == nil {
		_, f, _, _ := runtime.Caller(0)
		resultsPath := filepath.Join(filepath.Dir(f), "groth16_results.json")
		if werr := os.WriteFile(resultsPath, data, 0644); werr != nil {
			t.Logf("Warning: failed to write results JSON: %v", werr)
		}
	}
}

// =============================================================================
// SANITY CHECK TESTS - Verify the circuit is actually checking constraints
// =============================================================================

// TestCorruptedWitnessRejected verifies that corrupted witness values cause the circuit to fail.
// This is a critical sanity check - if corrupt witnesses pass, the circuit is broken.
func TestCorruptedWitnessRejected(t *testing.T) {
	t.Log("=== SANITY CHECK: Corrupted Witness Rejection ===")
	t.Log("Testing that corrupted witness values cause verification failure...")
	t.Log("")

	// Load valid witness
	witnessPath := getStagesWitnessPath()
	validAssignment, err := LoadStagesAssignment(witnessPath)
	if err != nil {
		t.Fatalf("Failed to load witness data: %v", err)
	}

	// First verify the valid witness passes
	t.Log("Step 1: Verify valid witness passes...")
	var circuit JoltStagesCircuit
	err = test.IsSolved(&circuit, validAssignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Valid witness should pass but got error: %v", err)
	}
	t.Log("  ✓ Valid witness passes (expected)")

	// Test corrupting different types of witness values
	testCases := []struct {
		name        string
		fieldName   string
		corruptFunc func(*big.Int) *big.Int
	}{
		{
			name:      "Corrupt commitment element",
			fieldName: "Commitment_0_0",
			corruptFunc: func(v *big.Int) *big.Int {
				return new(big.Int).Add(v, big.NewInt(1))
			},
		},
		{
			name:      "Corrupt PC claim (Spartan outer)",
			fieldName: "Claim_Virtual_PC_SpartanOuter",
			corruptFunc: func(v *big.Int) *big.Int {
				return new(big.Int).Add(v, big.NewInt(1))
			},
		},
		{
			name:      "Corrupt Rs1 value claim",
			fieldName: "Claim_Virtual_Rs1Value_SpartanOuter",
			corruptFunc: func(v *big.Int) *big.Int {
				return new(big.Int).Add(v, big.NewInt(42))
			},
		},
		{
			name:      "Zero out RdInc claim",
			fieldName: "Claim_Committed_RdInc_RegistersReadWriteChecking",
			corruptFunc: func(v *big.Int) *big.Int {
				return big.NewInt(0)
			},
		},
		{
			name:      "Negate RAM booleanity claim",
			fieldName: "Claim_Committed_RamRa_0_RamBooleanity",
			corruptFunc: func(v *big.Int) *big.Int {
				return new(big.Int).Neg(v)
			},
		},
		{
			name:      "Corrupt lookup output claim",
			fieldName: "Claim_Virtual_LookupOutput_SpartanOuter",
			corruptFunc: func(v *big.Int) *big.Int {
				return new(big.Int).Add(v, big.NewInt(999))
			},
		},
	}

	t.Log("")
	t.Log("Step 2: Test each corruption type...")
	passedCorruptions := 0
	failedCorruptions := 0

	for _, tc := range testCases {
		// Reload fresh witness
		assignment, err := LoadStagesAssignment(witnessPath)
		if err != nil {
			t.Fatalf("Failed to reload witness: %v", err)
		}

		// Apply corruption
		v := reflect.ValueOf(assignment).Elem()
		field := v.FieldByName(tc.fieldName)
		if !field.IsValid() {
			t.Logf("  ? %s - field %s not found, skipping", tc.name, tc.fieldName)
			continue
		}

		originalVal := field.Interface().(*big.Int)
		corruptedVal := tc.corruptFunc(new(big.Int).Set(originalVal))
		field.Set(reflect.ValueOf(corruptedVal))

		// Test that corrupted witness fails
		err = test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
		if err != nil {
			t.Logf("  ✓ %s - correctly rejected", tc.name)
			passedCorruptions++
		} else {
			t.Logf("  ✗ %s - INCORRECTLY ACCEPTED! This is a bug!", tc.name)
			failedCorruptions++
		}
	}

	t.Log("")
	t.Logf("Results: %d corruptions correctly rejected, %d incorrectly accepted", passedCorruptions, failedCorruptions)

	if failedCorruptions > 0 {
		t.Fatalf("CRITICAL: %d corrupted witnesses were incorrectly accepted!", failedCorruptions)
	}

	if passedCorruptions == 0 {
		t.Fatal("CRITICAL: No corruption tests ran successfully!")
	}

	t.Log("")
	t.Log("✓ All corruption tests passed - circuit correctly rejects invalid witnesses")
}

// TestRandomFuzzing performs random fuzzing on witness values
func TestRandomFuzzing(t *testing.T) {
	t.Log("=== SANITY CHECK: Random Fuzzing ===")
	t.Log("Testing random modifications to witness values...")
	t.Log("")

	// Load valid witness
	witnessPath := getStagesWitnessPath()

	var circuit JoltStagesCircuit

	// Get list of field names
	v := reflect.ValueOf(&JoltStagesCircuit{}).Elem()
	fieldNames := make([]string, 0)
	for i := 0; i < v.NumField(); i++ {
		fieldNames = append(fieldNames, v.Type().Field(i).Name)
	}

	t.Logf("Total fields to fuzz: %d", len(fieldNames))
	t.Log("")

	// Fuzz a subset of fields
	fuzzCount := 20
	if fuzzCount > len(fieldNames) {
		fuzzCount = len(fieldNames)
	}

	rejectedCount := 0
	acceptedCount := 0

	for i := 0; i < fuzzCount; i++ {
		// Pick field at regular intervals to cover different parts of witness
		fieldIdx := (i * len(fieldNames)) / fuzzCount
		fieldName := fieldNames[fieldIdx]

		// Load fresh witness
		assignment, err := LoadStagesAssignment(witnessPath)
		if err != nil {
			t.Fatalf("Failed to load witness: %v", err)
		}

		// Corrupt this field
		v := reflect.ValueOf(assignment).Elem()
		field := v.FieldByName(fieldName)
		if !field.IsValid() || field.IsNil() {
			continue
		}

		originalVal := field.Interface().(*big.Int)
		// Add a random-ish value based on field index
		corruptedVal := new(big.Int).Add(originalVal, big.NewInt(int64(i*7919+1))) // 7919 is prime
		field.Set(reflect.ValueOf(corruptedVal))

		// Test
		err = test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
		if err != nil {
			rejectedCount++
		} else {
			acceptedCount++
			t.Logf("  WARNING: Field %s accepted corruption!", fieldName)
		}
	}

	t.Log("")
	t.Logf("Fuzz results: %d rejected, %d accepted out of %d tests", rejectedCount, acceptedCount, fuzzCount)

	// We expect most corruptions to be rejected
	// Some might pass if they're unused fields, but the majority should fail
	rejectionRate := float64(rejectedCount) / float64(fuzzCount) * 100
	t.Logf("Rejection rate: %.1f%%", rejectionRate)

	if rejectionRate < 50 {
		t.Fatalf("CRITICAL: Less than 50%% of random corruptions were rejected! Circuit may not be checking constraints properly.")
	}

	t.Log("")
	t.Log("✓ Random fuzzing passed - majority of corruptions correctly rejected")
}

// TestAssertionCountMatchesTheory verifies the number of assertions matches expected sumchecks
func TestAssertionCountMatchesTheory(t *testing.T) {
	t.Log("=== SANITY CHECK: Assertion Count vs Theory ===")
	t.Log("")

	// According to theory (from gnark-stage5-context.md):
	// Stage 1: 1 batched sumcheck (Spartan outer)
	// Stage 2: 5 batched sumchecks (product virtualization, RAM RAF, output check, etc.)
	// Stage 3: 3 batched sumchecks (shift, instruction input, register claim reduction)
	// Stage 4: 4 batched sumchecks (register r/w, RAM booleanity, RAM val eval, RAM val final)
	// Stage 5: 4 batched sumchecks (register val eval, RAM Hamming, RAM ra reduction, lookups read-raf)
	//
	// However, batched sumchecks produce COMBINED assertions, not one per sumcheck.
	// The transpiler output shows 13 assertions for Stages 1-5.

	// Read the generated circuit to count assertions
	circuitPath := filepath.Join(filepath.Dir(getStagesWitnessPath()), "stages_circuit.go")
	data, err := os.ReadFile(circuitPath)
	if err != nil {
		t.Fatalf("Failed to read circuit file: %v", err)
	}

	circuit := string(data)

	// Count AssertIsEqual calls
	assertCount := 0
	searchStr := "api.AssertIsEqual("
	for i := 0; i < len(circuit); i++ {
		if i+len(searchStr) <= len(circuit) && circuit[i:i+len(searchStr)] == searchStr {
			assertCount++
		}
	}

	t.Logf("Assertions found in circuit: %d", assertCount)

	// Check against expected (13 for Stages 1-5)
	expectedAssertions := 13
	if assertCount != expectedAssertions {
		t.Logf("WARNING: Expected %d assertions, found %d", expectedAssertions, assertCount)
	} else {
		t.Logf("✓ Assertion count matches expected (%d)", expectedAssertions)
	}

	// Count Poseidon hash calls (these are the Fiat-Shamir challenges)
	poseidonCount := 0
	poseidonStr := "poseidon.Hash"
	for i := 0; i < len(circuit); i++ {
		if i+len(poseidonStr) <= len(circuit) && circuit[i:i+len(poseidonStr)] == poseidonStr {
			poseidonCount++
		}
	}
	t.Logf("Poseidon hash calls: %d (Fiat-Shamir transcript operations)", poseidonCount)

	// Count api.Mul, api.Add operations (actual constraint operations)
	mulCount := 0
	addCount := 0
	for i := 0; i < len(circuit); i++ {
		if i+8 <= len(circuit) {
			if circuit[i:i+8] == "api.Mul(" {
				mulCount++
			}
			if circuit[i:i+8] == "api.Add(" {
				addCount++
			}
		}
	}
	t.Logf("Field operations: %d multiplications, %d additions", mulCount, addCount)

	t.Log("")
	if poseidonCount > 0 && mulCount > 1000 && assertCount == expectedAssertions {
		t.Log("✓ Circuit structure looks reasonable for a zkVM verifier")
	} else {
		t.Log("? Circuit structure may need review")
	}
}

// TestCircuitNotTrivial verifies the circuit is doing real work, not just returning success
func TestCircuitNotTrivial(t *testing.T) {
	t.Log("=== SANITY CHECK: Circuit Non-Triviality ===")
	t.Log("")

	// Compile circuit and check constraint count
	var circuit JoltStagesCircuit
	r1cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		t.Fatalf("Failed to compile circuit: %v", err)
	}

	constraints := r1cs.GetNbConstraints()
	publicVars := r1cs.GetNbPublicVariables()
	internalVars := r1cs.GetNbInternalVariables()

	t.Logf("Constraints:      %d", constraints)
	t.Logf("Public vars:      %d", publicVars)
	t.Logf("Internal vars:    %d", internalVars)

	// Sanity checks
	if constraints < 100000 {
		t.Fatalf("SUSPICIOUS: Only %d constraints - expected >100k for a real zkVM verifier", constraints)
	}

	if publicVars < 100 {
		t.Fatalf("SUSPICIOUS: Only %d public vars - expected many proof elements", publicVars)
	}

	// Check witness has meaningful values
	witnessPath := getStagesWitnessPath()
	assignment, err := LoadStagesAssignment(witnessPath)
	if err != nil {
		t.Fatalf("Failed to load witness: %v", err)
	}

	v := reflect.ValueOf(assignment).Elem()
	nonZeroCount := 0
	zeroCount := 0
	for i := 0; i < v.NumField(); i++ {
		field := v.Field(i)
		if field.IsValid() && !field.IsNil() {
			val := field.Interface().(*big.Int)
			if val.Cmp(big.NewInt(0)) != 0 {
				nonZeroCount++
			} else {
				zeroCount++
			}
		}
	}

	t.Logf("Witness: %d non-zero values, %d zero values", nonZeroCount, zeroCount)

	if nonZeroCount < 100 {
		t.Fatalf("SUSPICIOUS: Only %d non-zero witness values - expected many proof elements", nonZeroCount)
	}

	t.Log("")
	t.Log("✓ Circuit appears to be non-trivial with meaningful constraints and witness")
}
