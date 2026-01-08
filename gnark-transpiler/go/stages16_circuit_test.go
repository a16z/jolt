package jolt_verifier

import (
	"testing"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

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
