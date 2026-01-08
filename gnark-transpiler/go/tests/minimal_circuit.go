package tests

import (
	"github.com/consensys/gnark/frontend"
	"jolt_verifier/poseidon"
)

type MinimalCircuit struct {
	X_0 frontend.Variable `gnark:",public"`
	X_1 frontend.Variable `gnark:",public"`
	X_2 frontend.Variable `gnark:",public"`
	X_3 frontend.Variable `gnark:",public"`
	Output frontend.Variable `gnark:",public"`
}

func (circuit *MinimalCircuit) Define(api frontend.API) error {
	result := api.Mul(api.Mul(api.Add(circuit.X_0, circuit.X_1), circuit.X_2), poseidon.Truncate128Reverse(api, poseidon.Hash(api, poseidon.Hash(api, poseidon.Hash(api, 30506420032924013, 0, 0), 0, poseidon.ByteReverse(api, circuit.X_3)), 1, 0)))
	api.AssertIsEqual(result, circuit.Output)
	return nil
}
