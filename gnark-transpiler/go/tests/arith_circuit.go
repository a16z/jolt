package tests

import "github.com/consensys/gnark/frontend"

type ArithCircuit struct {
	X_0 frontend.Variable `gnark:",public"`
	X_1 frontend.Variable `gnark:",public"`
	X_2 frontend.Variable `gnark:",public"`
	X_3 frontend.Variable `gnark:",public"`
	Output frontend.Variable `gnark:",public"`
}

func (circuit *ArithCircuit) Define(api frontend.API) error {
	result := api.Mul(api.Mul(api.Add(circuit.X_0, circuit.X_1), circuit.X_2), circuit.X_3)
	api.AssertIsEqual(result, circuit.Output)
	return nil
}
