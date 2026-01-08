package tests

import (
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

func TestArithCircuit(t *testing.T) {
	// (3 + 7) * 5 * 42 = 2100
	assignment := &ArithCircuit{
		X_0:    3,
		X_1:    7,
		X_2:    5,
		X_3:    42,
		Output: 2100,
	}

	var circuit ArithCircuit
	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		t.Fatalf("compile failed: %v", err)
	}
	t.Logf("Constraints: %d", cs.GetNbConstraints())

	pk, vk, err := groth16.Setup(cs)
	if err != nil {
		t.Fatalf("setup failed: %v", err)
	}

	witness, err := frontend.NewWitness(assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("witness failed: %v", err)
	}

	proof, err := groth16.Prove(cs, pk, witness)
	if err != nil {
		t.Fatalf("prove failed: %v", err)
	}

	publicWitness, _ := witness.Public()
	err = groth16.Verify(proof, vk, publicWitness)
	if err != nil {
		t.Fatalf("verify failed: %v", err)
	}

	t.Log("SUCCESS: Pure arithmetic circuit verified with Groth16!")
}
