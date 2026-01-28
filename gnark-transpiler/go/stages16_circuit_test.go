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

// getStages16WitnessPath returns the path to stages16_witness.json
func getStages16WitnessPath() string {
	_, currentFile, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(currentFile), "stages16_witness.json")
}

// LoadStages16Assignment loads witness data and creates a circuit assignment
func LoadStages16Assignment(witnessPath string) (*JoltStages16Circuit, error) {
	data, err := os.ReadFile(witnessPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read witness file: %w", err)
	}

	var witnessMap map[string]string
	if err := json.Unmarshal(data, &witnessMap); err != nil {
		return nil, fmt.Errorf("failed to parse witness JSON: %w", err)
	}

	assignment := &JoltStages16Circuit{}
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

	fmt.Printf("  Loaded %d witness values out of %d fields\n", loaded, v.NumField())
	return assignment, nil
}

func TestStages16CircuitCompile(t *testing.T) {
	t.Log("Jolt Stages 1-6 Verifier - Gnark Circuit Compilation Test")
	t.Log("")

	// Compile
	t.Log("Compiling circuit...")
	startCompile := time.Now()

	var circuit JoltStages16Circuit
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

func TestStages16CircuitSolver(t *testing.T) {
	t.Log("Jolt Stages 1-6 Verifier - Gnark Solver Debug Test")
	t.Log("")

	// Load witness
	witnessPath := getStages16WitnessPath()
	assignment, err := LoadStages16Assignment(witnessPath)
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
	var circuit JoltStages16Circuit
	err = test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Logf("Solver error: %v", err)
		// The error message should tell us which constraint failed
	} else {
		t.Log("All constraints satisfied!")
	}
}

func TestDebugA1Assertion(t *testing.T) {
	t.Log("Debug a1 assertion failure - tracing expected vs actual output claim")
	t.Log("")

	// Load witness
	witnessPath := getStages16WitnessPath()
	assignment, err := LoadStages16Assignment(witnessPath)
	if err != nil {
		t.Fatalf("Failed to load witness data: %v", err)
	}
	t.Logf("Loaded witness from: %s", witnessPath)

	v := reflect.ValueOf(assignment).Elem()
	p, _ := new(big.Int).SetString("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10)

	// Get the UnivariateSkip claim (input claim for sumcheck)
	inputClaim := v.FieldByName("Claim_Virtual_UnivariateSkip_SpartanOuter").Interface().(*big.Int)
	t.Logf("Input Claim (UnivariateSkip): %s", inputClaim.String())

	// To debug a1, we need to trace through the sumcheck verification
	// For degree-2 compressed polynomial:
	// coeffs_except_linear_term = [c0, c1, c2] where c0 = p(0), c1 = p(2), c2 = p(3)
	// The hint is p(1)
	// Linear term is recovered: linear = hint - 2*c0 - c1 - c2
	// Then p(r) = c0 + linear*r + c1*r^2 + c2*r^3

	// Wait - looking at outer.rs, this is degree 2, not degree 3
	// For degree-2: coeffs_except_linear_term = [c0, c2] where c0 = p(0), c2 = p(2)
	// linear = hint - 2*c0 - c2
	// p(r) = c0 + linear*r + c2*r^2

	// Actually let's look at what the circuit does more carefully
	// The sumcheck rounds have R{n}_0, R{n}_1, R{n}_2 which are:
	// R{n}_0 = p(0)
	// R{n}_1 = probably p(2) - p(0) or something similar
	// R{n}_2 = another coefficient

	// From the circuit line 837:
	// hint - R0_0 - R0_0 - R0_1 - R0_2
	// This means: linear_term = hint - 2*R0 - R1 - R2
	// Then p(r) = R0 + linear*r + R1*r^2 + R2*r^3?

	// No wait - looking at the formula in the circuit:
	// api.Sub(api.Sub(api.Sub(api.Sub(api.Mul(claim, 1), cse_35), R0_0), R0_0), R0_1), R0_2)
	// This is: claim*cse_35 - R0 - R0 - R1 - R2 = claim*cse_35 - 2*R0 - R1 - R2
	// Then it's multiplied by r (cse_34) and added to R0

	// Let me print all round data for inspection
	t.Log("\n=== Sumcheck Round Data ===")
	for round := 0; round <= 10; round++ {
		r0Name := fmt.Sprintf("Stage1_Sumcheck_R%d_0", round)
		r1Name := fmt.Sprintf("Stage1_Sumcheck_R%d_1", round)
		r2Name := fmt.Sprintf("Stage1_Sumcheck_R%d_2", round)

		r0 := v.FieldByName(r0Name).Interface().(*big.Int)
		r1 := v.FieldByName(r1Name).Interface().(*big.Int)
		r2 := v.FieldByName(r2Name).Interface().(*big.Int)

		t.Logf("Round %d: R0=%s, R1=%s, R2=%s", round, r0.String(), r1.String(), r2.String())
	}

	// Print all virtual polynomial claims that appear in the expected output computation
	t.Log("\n=== Virtual Polynomial Claims ===")
	claimNames := []string{
		"Claim_Virtual_OpFlags_Load_SpartanOuter",
		"Claim_Virtual_OpFlags_Store_SpartanOuter",
		"Claim_Virtual_OpFlags_AddOperands_SpartanOuter",
		"Claim_Virtual_OpFlags_SubtractOperands_SpartanOuter",
		"Claim_Virtual_OpFlags_MultiplyOperands_SpartanOuter",
		"Claim_Virtual_OpFlags_Assert_SpartanOuter",
		"Claim_Virtual_ShouldJump_SpartanOuter",
		"Claim_Virtual_OpFlags_VirtualInstruction_SpartanOuter",
		"Claim_Virtual_NextIsVirtual_SpartanOuter",
		"Claim_Virtual_NextIsFirstInSequence_SpartanOuter",
		"Claim_Virtual_RamAddress_SpartanOuter",
		"Claim_Virtual_RamReadValue_SpartanOuter",
		"Claim_Virtual_RamWriteValue_SpartanOuter",
		"Claim_Virtual_RdWriteValue_SpartanOuter",
		"Claim_Virtual_Rs2Value_SpartanOuter",
		"Claim_Virtual_LeftLookupOperand_SpartanOuter",
		"Claim_Virtual_LeftInstructionInput_SpartanOuter",
		"Claim_Virtual_LookupOutput_SpartanOuter",
		"Claim_Virtual_NextUnexpandedPC_SpartanOuter",
		"Claim_Virtual_PC_SpartanOuter",
		"Claim_Virtual_OpFlags_DoNotUpdateUnexpandedPC_SpartanOuter",
		"Claim_Virtual_Rs1Value_SpartanOuter",
		"Claim_Virtual_Imm_SpartanOuter",
		"Claim_Virtual_RightLookupOperand_SpartanOuter",
		"Claim_Virtual_RightInstructionInput_SpartanOuter",
		"Claim_Virtual_Product_SpartanOuter",
		"Claim_Virtual_UnexpandedPC_SpartanOuter",
		"Claim_Virtual_OpFlags_IsCompressed_SpartanOuter",
		"Claim_Virtual_ShouldBranch_SpartanOuter",
		"Claim_Virtual_OpFlags_Jump_SpartanOuter",
		"Claim_Virtual_WriteLookupOutputToRD_SpartanOuter",
		"Claim_Virtual_WritePCtoRD_SpartanOuter",
		"Claim_Virtual_OpFlags_Advice_SpartanOuter",
		"Claim_Virtual_NextPC_SpartanOuter",
	}

	claimsFound := 0
	for _, name := range claimNames {
		field := v.FieldByName(name)
		if field.IsValid() && !field.IsZero() {
			val := field.Interface().(*big.Int)
			// Check if value is non-zero
			if val.Cmp(big.NewInt(0)) != 0 {
				t.Logf("  %s = %s", name, val.String())
				claimsFound++
			}
		}
	}
	t.Logf("Found %d non-zero claims", claimsFound)

	// Now let's manually compute the sumcheck output claim step by step
	// This matches the circuit formula in a1
	t.Log("\n=== Manual Sumcheck Output Claim Computation ===")

	// We need the challenges and the batching coefficient
	// The challenges are derived from Poseidon hashes, which we can't compute here
	// But we can check if the witness values are consistent with themselves

	// The sumcheck verification starts with inputClaim and applies 11 rounds
	// Each round: e = e(r) where e is the compressed polynomial
	// The formula is complex because of compressed polynomial representation

	// Actually the issue is we need the challenge values, which are computed
	// via Poseidon in the circuit. We don't have direct access to them in the witness.

	// Let me check if there are challenge values in the witness
	t.Log("\n=== Looking for challenge-related witness values ===")
	tt := v.Type()
	for i := 0; i < v.NumField(); i++ {
		fieldName := tt.Field(i).Name
		if len(fieldName) > 0 {
			// Look for r_tail or challenge-related fields
			if len(fieldName) > 5 && fieldName[:5] == "R_Tai" {
				field := v.Field(i)
				if field.IsValid() && !field.IsZero() {
					t.Logf("  %s = %v", fieldName, field.Interface())
				}
			}
		}
	}

	// The key insight: the expected output claim formula involves:
	// 1. tau_high_bound_r0 - a Lagrange basis evaluation
	// 2. tau_bound_r_tail - EQ polynomial over r_tail challenges
	// 3. inner_sum_prod - the R1CS inner sum product

	// Since we can't easily compute these without the challenges,
	// let's at least verify the sumcheck polynomial consistency

	t.Log("\n=== Sumcheck Polynomial Consistency Check ===")
	// For a valid sumcheck round, p(0) + p(1) should equal the current claim
	// p(0) = R_0, p(1) = hint, so R_0 + hint = current claim

	// But we need the hint values, which aren't directly in the witness...
	// The hint is computed as: 2*R0 + R1 + R2 + linear_term
	// where linear_term = hint - 2*R0 - R1 - R2
	// This is circular, so we need the actual hint values

	// Let's verify: p(0) + p(1) = claim for round 0
	// p(0) = R0_0
	// p(1) = hint = claim - p(0) if sumcheck is valid (since p(0) + p(1) = claim)
	// So hint = inputClaim - R0_0

	r0_0 := v.FieldByName("Stage1_Sumcheck_R0_0").Interface().(*big.Int)
	expectedHint0 := new(big.Int).Sub(inputClaim, r0_0)
	expectedHint0.Mod(expectedHint0, p)
	t.Logf("Round 0: R0_0 = %s", r0_0.String())
	t.Logf("Round 0: Expected hint (inputClaim - R0_0) = %s", expectedHint0.String())

	// Check a key value: the claimed output
	t.Log("\n=== Tracing the Sumcheck Output vs Expected ===")

	// The sumcheck output claim is computed iteratively through 11 rounds
	// Starting from inputClaim * batchingCoeff, then evaluating compressed poly at each challenge

	// Compressed poly evaluation: p(r) = c0 + linear*r + c1*r^2 + c2*r^3
	// where linear = hint - 2*c0 - c1 - c2
	// and hint = p(1) which satisfies p(0) + p(1) = previous_claim

	// Let's manually compute the sumcheck output
	// We need the challenges, which are derived from Poseidon hashes
	// Unfortunately we don't have direct access to them in the witness

	// Instead, let's verify the constraint formula structure
	// The hint for round 0 is: inputClaim - R0_0 (since p(0) + p(1) = inputClaim)
	// where inputClaim is batching_coeff * claim_after_uni_skip
	// But wait, in the single-instance case, there's still a batching coeff!

	// Let me print what we know
	t.Logf("Input Claim (UnivariateSkip): %s", inputClaim.String())
	t.Logf("Round 0: R0_0 (p(0)) = %s", r0_0.String())

	// The initial claim for the batched sumcheck should be:
	// initial_claim = input_claim * batching_coeff
	// And the first round's hint is: initial_claim - R0_0
	// which should equal R0_0 + R0_1 + R0_2 (since p(0)+p(1)+p(2)+p(3) = initial_claim for degree 3)

	// Actually for degree 2: p(0) + p(1) = claim
	// So hint = claim - p(0)

	// Let me check if there's a pattern in the sumcheck values
	t.Log("\n=== Checking Sumcheck Round Consistency ===")
	for round := 0; round <= 10; round++ {
		r0Name := fmt.Sprintf("Stage1_Sumcheck_R%d_0", round)
		r1Name := fmt.Sprintf("Stage1_Sumcheck_R%d_1", round)
		r2Name := fmt.Sprintf("Stage1_Sumcheck_R%d_2", round)

		r0 := v.FieldByName(r0Name).Interface().(*big.Int)
		r1 := v.FieldByName(r1Name).Interface().(*big.Int)
		r2 := v.FieldByName(r2Name).Interface().(*big.Int)

		// For a degree-2 compressed poly: [c0, c2] where c0=p(0), c2=p(2)
		// The hint p(1) should satisfy p(0) + p(1) = claim
		// p(2) is stored separately

		// Sum of evaluations at 0, 1, 2 (but we only have c0 and c2)
		// Let's just print the sum c0 + c2
		sum := new(big.Int).Add(r0, r1)
		sum.Add(sum, r2)
		sum.Mod(sum, p)

		t.Logf("Round %d: R0+R1+R2 (mod p) = %s", round, sum.String())
	}

	// Use gnark test solver to check constraints
	var circuit JoltStages16Circuit
	err = test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Logf("Solver error: %v", err)
	} else {
		t.Log("All constraints satisfied!")
	}
}

func TestStages16CircuitProveVerify(t *testing.T) {
	t.Log("Jolt Stages 1-6 Verifier - Full Groth16 Prove/Verify Test")
	t.Log("")

	// Load witness
	witnessPath := getStages16WitnessPath()
	assignment, err := LoadStages16Assignment(witnessPath)
	if err != nil {
		t.Fatalf("Failed to load witness data: %v", err)
	}
	t.Logf("Loaded witness from: %s", witnessPath)

	// Compile
	t.Log("")
	t.Log("Compiling circuit...")
	startCompile := time.Now()

	var circuit JoltStages16Circuit
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
	t.Log("✓ Stages 1-6 circuit verification passed!")
}
