package jolt_verifier

import (
	"bytes"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

// getWitnessDataPath returns the path to witness_data.json
func getWitnessDataPath() string {
	_, currentFile, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(currentFile), "..", "data", "witness_data.json")
}

func TestStage1Circuit(t *testing.T) {
	t.Log("Jolt Stage 1 Verifier - Gnark Circuit Test")
	t.Log("")

	// Load witness
	witnessPath := getWitnessDataPath()
	witnessData, err := LoadWitnessData(witnessPath)

	if err != nil {
		t.Fatalf("Failed to load witness data: %v", err)
	}

	assignment, err := LoadStage1Assignment(witnessPath)
	if err != nil {
		t.Fatalf("Failed to create assignment: %v", err)
	}

	totalInputs := len(witnessData.Commitments)*12 + len(witnessData.UniSkipCoeffs) + len(witnessData.SumcheckPolys)*3 + len(witnessData.R1csInputEvals)
	t.Logf("Witness: %d commitments, %d uni-skip coeffs, %d sumcheck rounds, %d r1cs inputs (%d total field elements)",
		len(witnessData.Commitments), len(witnessData.UniSkipCoeffs), len(witnessData.SumcheckPolys), len(witnessData.R1csInputEvals), totalInputs)

	// Compile
	t.Log("")
	t.Log("Compiling circuit...")
	startCompile := time.Now()

	var circuit Stage1Circuit
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
	t.Log("✓ Stage 1 circuit verification passed!")
}
