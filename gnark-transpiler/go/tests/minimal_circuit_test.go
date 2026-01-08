package tests

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

func TestMinimalCircuit(t *testing.T) {
	// Expected output from Rust computation
	expected := new(big.Int)
	expected.SetString("950168490859324664956783777980982594150", 10)

	assignment := &MinimalCircuit{
		X_0:    3,  // a
		X_1:    7,  // b
		X_2:    5,  // c
		X_3:    42, // d
		Output: expected,
	}

	var circuit MinimalCircuit
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

	t.Log("SUCCESS: Minimal circuit verified with Groth16!")
}
