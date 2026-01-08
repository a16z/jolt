package tests

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

func TestByteReverseCircuit(t *testing.T) {
	// byte_reverse(42) = 18997139640497188311679614727987859882177106859206655037723509876298247831552
	expected := new(big.Int)
	expected.SetString("18997139640497188311679614727987859882177106859206655037723509876298247831552", 10)

	assignment := &ByteReverseCircuit{
		X_0:    42,
		Output: expected,
	}

	var circuit ByteReverseCircuit
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

	t.Log("SUCCESS: ByteReverse circuit verified with Groth16!")
}
