package tests

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
)

func TestPoseidonCircuit(t *testing.T) {
	// poseidon(3, 7, 5) = 20741747683121754121593915476606842481422161480605326911875053857442610499000
	expected := new(big.Int)
	expected.SetString("20741747683121754121593915476606842481422161480605326911875053857442610499000", 10)

	assignment := &PoseidonCircuit{
		X_0:    3,
		X_1:    7,
		X_2:    5,
		Output: expected,
	}

	var circuit PoseidonCircuit
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

	t.Log("SUCCESS: Pure Poseidon circuit verified with Groth16!")
}
