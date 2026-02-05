package jolt_verifier

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/test"
)

func TestMinimalCircuit(t *testing.T) {
	// Values from Rust computation:
	// Inputs: a=3, b=7, c=5, d=42
	// Expected output = 5246902962866244651916644613801965761300
	assert := test.NewAssert(t)

	var circuit MinimalCircuit

	expected := new(big.Int)
	expected.SetString("5246902962866244651916644613801965761300", 10)

	assignment := &MinimalCircuit{
		X_0:    3,
		X_1:    7,
		X_2:    5,
		X_3:    42,
		Output: expected,
	}

	assert.ProverSucceeded(&circuit, assignment, test.WithCurves(ecc.BN254))
}

// DebugMinimalCircuit is a version that exposes intermediate hash values
type DebugMinimalCircuit struct {
	X_3             frontend.Variable `gnark:",public"`
	ExpectedHash1   frontend.Variable `gnark:",public"` // Hash(label, 0, 0)
	ExpectedHash2   frontend.Variable `gnark:",public"` // Hash(prev, 0, 100)
	ExpectedHash3   frontend.Variable `gnark:",public"` // Hash(prev, 1, ByteReverse(X_3))
	ExpectedHash4   frontend.Variable `gnark:",public"` // Hash(prev, 2, 0)
	ExpectedTrunc   frontend.Variable `gnark:",public"` // Truncate128(Hash4)
}

func (circuit *DebugMinimalCircuit) Define(api frontend.API) error {
	// Import poseidon package
	api.Compiler()

	// These use the jolt_verifier/poseidon package
	// Minimal imports
	return nil
}
