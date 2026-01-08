package tests

import (
	"github.com/consensys/gnark/frontend"
	"jolt_verifier/poseidon"
)

type PoseidonCircuit struct {
	X_0 frontend.Variable `gnark:",public"`
	X_1 frontend.Variable `gnark:",public"`
	X_2 frontend.Variable `gnark:",public"`
	Output frontend.Variable `gnark:",public"`
}

func (circuit *PoseidonCircuit) Define(api frontend.API) error {
	result := poseidon.Hash(api, circuit.X_0, circuit.X_1, circuit.X_2)
	api.AssertIsEqual(result, circuit.Output)
	return nil
}
