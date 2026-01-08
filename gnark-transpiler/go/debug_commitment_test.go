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

// Circuit to verify commitment hashing
// Tests: state after preamble + first commitment
type DebugCommitmentCircuit struct {
	// State after preamble (known good)
	StateAfterPreamble frontend.Variable `gnark:",public"`
	// First commitment chunks (12 values)
	Commitment0Chunk0  frontend.Variable `gnark:",public"`
	Commitment0Chunk1  frontend.Variable `gnark:",public"`
	Commitment0Chunk2  frontend.Variable `gnark:",public"`
	Commitment0Chunk3  frontend.Variable `gnark:",public"`
	Commitment0Chunk4  frontend.Variable `gnark:",public"`
	Commitment0Chunk5  frontend.Variable `gnark:",public"`
	Commitment0Chunk6  frontend.Variable `gnark:",public"`
	Commitment0Chunk7  frontend.Variable `gnark:",public"`
	Commitment0Chunk8  frontend.Variable `gnark:",public"`
	Commitment0Chunk9  frontend.Variable `gnark:",public"`
	Commitment0Chunk10 frontend.Variable `gnark:",public"`
	Commitment0Chunk11 frontend.Variable `gnark:",public"`
	// Expected state after first commitment (from Rust)
	ExpectedState frontend.Variable `gnark:",public"`
}

func (c *DebugCommitmentCircuit) Define(api frontend.API) error {
	// Start with state after preamble
	state := c.StateAfterPreamble

	// Commitment hashing: first chunk uses n_rounds=8, rest use 0
	// This matches the circuit structure from stage1_circuit.go
	zero := frontend.Variable(0)
	nRounds := frontend.Variable(8) // n_rounds after preamble

	// Hash first chunk with n_rounds
	state = poseidon.Hash(api, state, nRounds, c.Commitment0Chunk0)

	// Hash remaining chunks with 0
	state = poseidon.Hash(api, state, zero, c.Commitment0Chunk1)
	state = poseidon.Hash(api, state, zero, c.Commitment0Chunk2)
	state = poseidon.Hash(api, state, zero, c.Commitment0Chunk3)
	state = poseidon.Hash(api, state, zero, c.Commitment0Chunk4)
	state = poseidon.Hash(api, state, zero, c.Commitment0Chunk5)
	state = poseidon.Hash(api, state, zero, c.Commitment0Chunk6)
	state = poseidon.Hash(api, state, zero, c.Commitment0Chunk7)
	state = poseidon.Hash(api, state, zero, c.Commitment0Chunk8)
	state = poseidon.Hash(api, state, zero, c.Commitment0Chunk9)
	state = poseidon.Hash(api, state, zero, c.Commitment0Chunk10)
	state = poseidon.Hash(api, state, zero, c.Commitment0Chunk11)

	// Verify state matches expected
	api.AssertIsEqual(state, c.ExpectedState)

	return nil
}

func TestDebugCommitment(t *testing.T) {
	// State after preamble (from Rust)
	// This is the state BEFORE commitments are hashed
	// We need to get this from running the Rust code
	stateAfterPreamble, _ := new(big.Int).SetString("15347626584322578373199561629687799511864240952792169390776318568321796642585", 10)

	// First commitment chunks from stage1_circuit_test.go
	chunk0, _ := new(big.Int).SetString("7485384150160967088772444117152036183843825293618005154080868084803912372013", 10)
	chunk1, _ := new(big.Int).SetString("16801368601274720200971535067107586575940911932244671518518875269208590998287", 10)
	chunk2, _ := new(big.Int).SetString("9995533289864093739239393658158480297291040875419283799521546297135696949524", 10)
	chunk3, _ := new(big.Int).SetString("3410368980813680732469783837790858171026433158680404348548964414744120212015", 10)
	chunk4, _ := new(big.Int).SetString("16725052492841063018502735087009266194136292694671347980475737651588874256130", 10)
	chunk5, _ := new(big.Int).SetString("21820697016754370326558791406063150887944545626397956844395781943785457852181", 10)
	chunk6, _ := new(big.Int).SetString("17893048976680079196808237929769640610242123723889763748364515333255027389438", 10)
	chunk7, _ := new(big.Int).SetString("9046684748698323878265788583025336441061444705522113914255594983765630725669", 10)
	chunk8, _ := new(big.Int).SetString("20414605890606527971522086518586027556702362497891099009289716186595842468906", 10)
	chunk9, _ := new(big.Int).SetString("3307560267472949867956713403593861428514049631078198197127428893162510369824", 10)
	chunk10, _ := new(big.Int).SetString("18388905656004717090776865885929351598081305442102263024210244038426091774764", 10)
	chunk11, _ := new(big.Int).SetString("1028728618743453121018242684564372213012797197345564149274510118410927201285", 10)

	// Expected state after first commitment (need to get from Rust)
	// For now, let's leave it as 0 and see what the circuit computes
	expectedState, _ := new(big.Int).SetString("0", 10)

	assignment := &DebugCommitmentCircuit{
		StateAfterPreamble: stateAfterPreamble,
		Commitment0Chunk0:  chunk0,
		Commitment0Chunk1:  chunk1,
		Commitment0Chunk2:  chunk2,
		Commitment0Chunk3:  chunk3,
		Commitment0Chunk4:  chunk4,
		Commitment0Chunk5:  chunk5,
		Commitment0Chunk6:  chunk6,
		Commitment0Chunk7:  chunk7,
		Commitment0Chunk8:  chunk8,
		Commitment0Chunk9:  chunk9,
		Commitment0Chunk10: chunk10,
		Commitment0Chunk11: chunk11,
		ExpectedState:      expectedState,
	}

	var circuit DebugCommitmentCircuit
	_, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}

	err = test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Logf("Commitment circuit not satisfied (expected with 0): %v", err)
		t.Log("Need to get expected state from Rust to verify")
	} else {
		t.Log("Commitment circuit passed!")
	}
}
