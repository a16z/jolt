package jolt_verifier

// This test is disabled - LoadStage1DebugAssignment function doesn't exist
// TODO: Implement LoadStage1DebugAssignment or remove this file

// import (
// 	"testing"
//
// 	"github.com/consensys/gnark-crypto/ecc"
// 	"github.com/consensys/gnark/frontend"
// 	"github.com/consensys/gnark/frontend/cs/r1cs"
// 	"github.com/consensys/gnark/test"
// )
//
// func TestStage1DebugCircuit(t *testing.T) {
// 	t.Log("Stage 1 Debug Circuit - Printing intermediate values")
// 	t.Log("")
//
// 	// Load witness
// 	witnessPath := getWitnessDataPath()
//
// 	assignment, err := LoadStage1DebugAssignment(witnessPath)
// 	if err != nil {
// 		t.Fatalf("Failed to create assignment: %v", err)
// 	}
//
// 	// Compile with debug prints
// 	t.Log("Compiling debug circuit...")
//
// 	var circuit Stage1DebugCircuit
// 	cs, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
// 	if err != nil {
// 		t.Fatalf("Failed to compile circuit: %v", err)
// 	}
//
// 	t.Logf("Compiled: %d constraints", cs.GetNbConstraints())
//
// 	// Solve the circuit - this triggers api.Println statements
// 	t.Log("")
// 	t.Log("Solving circuit (debug values will be printed)...")
// 	t.Log("")
//
// 	// Use test.IsSolved which internally runs the solver and triggers Println
// 	err = test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
// 	if err != nil {
// 		t.Fatalf("Circuit not satisfied: %v", err)
// 	}
//
// 	t.Log("")
// 	t.Log("Debug circuit executed successfully")
// }
