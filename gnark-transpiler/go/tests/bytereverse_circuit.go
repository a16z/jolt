package tests

import (
	"github.com/consensys/gnark/frontend"
	"jolt_verifier/poseidon"
)

type ByteReverseCircuit struct {
	X_0 frontend.Variable `gnark:",public"`
	Output frontend.Variable `gnark:",public"`
}

func (circuit *ByteReverseCircuit) Define(api frontend.API) error {
	result := poseidon.ByteReverse(api, circuit.X_0)
	api.AssertIsEqual(result, circuit.Output)
	return nil
}
