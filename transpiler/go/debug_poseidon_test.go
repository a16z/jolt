package jolt_verifier

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/test"

	poseidon "jolt_verifier/poseidon"
)

// Circuit to verify complete preamble
type DebugPreambleCircuit struct {
	// Expected final state after preamble (from Rust)
	ExpectedFinalState frontend.Variable `gnark:",public"`
}

func (c *DebugPreambleCircuit) Define(api frontend.API) error {
	// Initial state: hash(label, 0, 0) where label = "test" = 1953719668
	labelVal := frontend.Variable(1953719668)
	zero := frontend.Variable(0)
	state := poseidon.Hash(api, labelVal, zero, zero)

	// Preamble operations (matching Rust exactly):
	// 1. append_u64(max_input_size=4096)
	transformed := poseidon.AppendU64Transform(api, 4096)
	state = poseidon.Hash(api, state, frontend.Variable(0), transformed)

	// 2. append_u64(max_output_size=4096)
	state = poseidon.Hash(api, state, frontend.Variable(1), transformed)

	// 3. append_u64(memory_size=32768)
	transformed32k := poseidon.AppendU64Transform(api, 32768)
	state = poseidon.Hash(api, state, frontend.Variable(2), transformed32k)

	// 4. append_bytes(inputs=[50])
	inputsVal := frontend.Variable(50)
	state = poseidon.Hash(api, state, frontend.Variable(3), inputsVal)

	// 5. append_bytes(outputs=[225, 242, 204, 241, 46])
	// outputs as LE = 225 + 242*256 + 204*256^2 + 241*256^3 + 46*256^4 = 201625236193
	outputsVal := frontend.Variable(201625236193)
	state = poseidon.Hash(api, state, frontend.Variable(4), outputsVal)

	// 6. append_u64(panic=0)
	transformedZero := poseidon.AppendU64Transform(api, 0)
	state = poseidon.Hash(api, state, frontend.Variable(5), transformedZero)

	// 7. append_u64(ram_k=8192)
	transformed8k := poseidon.AppendU64Transform(api, 8192)
	state = poseidon.Hash(api, state, frontend.Variable(6), transformed8k)

	// 8. append_u64(trace_length=1024)
	transformed1k := poseidon.AppendU64Transform(api, 1024)
	state = poseidon.Hash(api, state, frontend.Variable(7), transformed1k)

	// Verify final state matches Rust
	api.AssertIsEqual(state, c.ExpectedFinalState)

	return nil
}

func TestDebugPreamble(t *testing.T) {
	// From Rust: Final state after preamble = 15347626584322578373199561629687799511864240952792169390776318568321796642585
	expectedFinal, _ := new(big.Int).SetString("15347626584322578373199561629687799511864240952792169390776318568321796642585", 10)

	assignment := &DebugPreambleCircuit{
		ExpectedFinalState: expectedFinal,
	}

	var circuit DebugPreambleCircuit
	_, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}

	err = test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Preamble circuit not satisfied: %v", err)
	}

	t.Log("✓ Complete preamble matches Rust!")
}
