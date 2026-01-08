package poseidon

import (
	"fmt"
	"math/big"
	"testing"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/test"
)

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

func TestPoseidonManual(t *testing.T) {
	// Print the first few constants to verify they match
	fmt.Println("=== Go Poseidon Constants ===")
	fmt.Println("cConstants[0] =", cConstants[0].String())
	fmt.Println("cConstants[1] =", cConstants[1].String())
	fmt.Println("cConstants[2] =", cConstants[2].String())
	fmt.Println("cConstants[3] =", cConstants[3].String())

	// Expected from light-poseidon circom t=4:
	// ark[0] = 11633431549750490989983886834189948010834808234699737327785600195936805266405
	expected := new(big.Int)
	expected.SetString("11633431549750490989983886834189948010834808234699737327785600195936805266405", 10)

	if cConstants[0].Cmp(expected) != 0 {
		t.Errorf("cConstants[0] mismatch!\nGot:      %s\nExpected: %s", cConstants[0].String(), expected.String())
	} else {
		fmt.Println("cConstants[0] matches light-poseidon!")
	}
}
