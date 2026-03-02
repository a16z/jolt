package poseidon

import (
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/test"
)

// Test vectors are generated from the Rust implementation using:
//   cargo run -p transpiler --release --bin poseidon_vectors
//
// This ensures the Go implementation matches the Rust light-poseidon circom parameters.
// If you modify the Poseidon implementation, regenerate test vectors with the command above.

// TestCircuit for testing Poseidon hash
type TestPoseidonCircuit struct {
	In1      frontend.Variable
	In2      frontend.Variable
	In3      frontend.Variable
	Expected frontend.Variable `gnark:",public"`
}

func (c *TestPoseidonCircuit) Define(api frontend.API) error {
	result := Hash(api, c.In1, c.In2, c.In3)
	api.AssertIsEqual(result, c.Expected)
	return nil
}

// TestPoseidonHashZeros verifies Poseidon hash of [0,0,0] matches Rust implementation.
// This validates the core hash function with the simplest input.
func TestPoseidonHashZeros(t *testing.T) {
	// Rust: hash([0, 0, 0]) = 5317387130258456662214331362918410991734007599705406860481038345552731150762
	assert := test.NewAssert(t)

	var circuit TestPoseidonCircuit

	expected := new(big.Int)
	expected.SetString("5317387130258456662214331362918410991734007599705406860481038345552731150762", 10)

	assignment := &TestPoseidonCircuit{
		In1:      0,
		In2:      0,
		In3:      0,
		Expected: expected,
	}

	assert.ProverSucceeded(&circuit, assignment, test.WithCurves(ecc.BN254))
}

// TestPoseidonHash123 verifies Poseidon hash of [1,2,3] matches Rust implementation.
// This tests the hash function with non-zero inputs to catch coefficient errors.
func TestPoseidonHash123(t *testing.T) {
	// Rust: hash([1, 2, 3]) = 6542985608222806190361240322586112750744169038454362455181422643027100751666
	assert := test.NewAssert(t)

	var circuit TestPoseidonCircuit

	expected := new(big.Int)
	expected.SetString("6542985608222806190361240322586112750744169038454362455181422643027100751666", 10)

	assignment := &TestPoseidonCircuit{
		In1:      1,
		In2:      2,
		In3:      3,
		Expected: expected,
	}

	assert.ProverSucceeded(&circuit, assignment, test.WithCurves(ecc.BN254))
}

// TestPoseidonConstants verifies the ark[0] constant matches light-poseidon circom t=4.
// This ensures the round constants are correctly imported from the reference implementation.
func TestPoseidonConstants(t *testing.T) {
	// Verify first constant matches light-poseidon circom t=4
	// ark[0] = 11633431549750490989983886834189948010834808234699737327785600195936805266405
	expected := new(big.Int)
	expected.SetString("11633431549750490989983886834189948010834808234699737327785600195936805266405", 10)

	if cConstants[0].Cmp(expected) != 0 {
		t.Errorf("cConstants[0] mismatch!\nGot:      %s\nExpected: %s", cConstants[0].String(), expected.String())
	}
}

