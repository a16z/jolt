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

// Circuit to test ByteReverse hint
type TestByteReverseCircuit struct {
	Input    frontend.Variable `gnark:",public"`
	Expected frontend.Variable `gnark:",public"`
}

func (c *TestByteReverseCircuit) Define(api frontend.API) error {
	result := poseidon.ByteReverse(api, c.Input)
	api.AssertIsEqual(result, c.Expected)
	return nil
}

// Circuit to test Truncate128Reverse hint
type TestTruncate128ReverseCircuit struct {
	Input    frontend.Variable `gnark:",public"`
	Expected frontend.Variable `gnark:",public"`
}

func (c *TestTruncate128ReverseCircuit) Define(api frontend.API) error {
	result := poseidon.Truncate128Reverse(api, c.Input)
	api.AssertIsEqual(result, c.Expected)
	return nil
}

func TestByteReverseHint(t *testing.T) {
	// From Rust: Input: 12345678901234567890
	// ByteReverse: 7450615490314221257910987457393031743665242432095373547377844877578360324092
	input, _ := new(big.Int).SetString("12345678901234567890", 10)
	expected, _ := new(big.Int).SetString("7450615490314221257910987457393031743665242432095373547377844877578360324092", 10)

	assignment := &TestByteReverseCircuit{
		Input:    input,
		Expected: expected,
	}

	var circuit TestByteReverseCircuit
	_, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}

	err = test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("ByteReverse hint mismatch: %v", err)
	}

	t.Log("ByteReverse hint matches Rust!")
}

func TestTruncate128ReverseHint(t *testing.T) {
	// From Rust (n_rounds = 98):
	// Hash result: 19654309276864454743090868671061718225298090476198580618438328100134481672469
	// r_sumcheck[0] = 6546053814691896674008619906195315247069748678507320726103401577775255757342
	input, _ := new(big.Int).SetString("19654309276864454743090868671061718225298090476198580618438328100134481672469", 10)
	expected, _ := new(big.Int).SetString("6546053814691896674008619906195315247069748678507320726103401577775255757342", 10)

	assignment := &TestTruncate128ReverseCircuit{
		Input:    input,
		Expected: expected,
	}

	var circuit TestTruncate128ReverseCircuit
	_, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}

	err = test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Truncate128Reverse hint mismatch: %v", err)
	}

	t.Log("Truncate128Reverse hint matches Rust!")
}

// Test sumcheck challenge computation using exact Rust values
type TestSumcheckChallengeCircuit struct {
	State         frontend.Variable `gnark:",public"`
	NRounds       frontend.Variable `gnark:",public"`
	ExpectedChallenge frontend.Variable `gnark:",public"`
}

func (c *TestSumcheckChallengeCircuit) Define(api frontend.API) error {
	// The challenge is: Truncate128Reverse(Hash(state, n_rounds, 0))
	hashResult := poseidon.Hash(api, c.State, c.NRounds, 0)
	challenge := poseidon.Truncate128Reverse(api, hashResult)
	api.AssertIsEqual(challenge, c.ExpectedChallenge)
	return nil
}

func TestSumcheckChallengeFromRustState(t *testing.T) {
	// From Rust debug_uniskip with correct n_rounds = 98:
	// State after round 0 poly: 17837831107584858481680424076579958046146299570891807727648586105247099570614
	// n_rounds = 98
	// Hash result = poseidon(state, 98, 0) = 19654309276864454743090868671061718225298090476198580618438328100134481672469
	// r_sumcheck[0] = 6546053814691896674008619906195315247069748678507320726103401577775255757342

	state, _ := new(big.Int).SetString("17837831107584858481680424076579958046146299570891807727648586105247099570614", 10)
	nRounds := 98
	expectedChallenge, _ := new(big.Int).SetString("6546053814691896674008619906195315247069748678507320726103401577775255757342", 10)

	// First, verify the hash
	expectedHash, _ := new(big.Int).SetString("19654309276864454743090868671061718225298090476198580618438328100134481672469", 10)
	t.Logf("State: %s", state.String())
	t.Logf("n_rounds: %d", nRounds)
	t.Logf("Expected hash: %s", expectedHash.String())
	t.Logf("Expected challenge: %s", expectedChallenge.String())

	assignment := &TestSumcheckChallengeCircuit{
		State:             state,
		NRounds:           nRounds,
		ExpectedChallenge: expectedChallenge,
	}

	var circuit TestSumcheckChallengeCircuit
	_, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}

	err = test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Sumcheck challenge mismatch: %v", err)
	}

	t.Log("✓ Sumcheck challenge matches Rust!")
}

// Test just the Poseidon hash (without truncation)
type TestPoseidonHashCircuit struct {
	State        frontend.Variable `gnark:",public"`
	NRounds      frontend.Variable `gnark:",public"`
	ExpectedHash frontend.Variable `gnark:",public"`
}

func (c *TestPoseidonHashCircuit) Define(api frontend.API) error {
	result := poseidon.Hash(api, c.State, c.NRounds, 0)
	api.AssertIsEqual(result, c.ExpectedHash)
	return nil
}

func TestPoseidonHashOnly(t *testing.T) {
	// From Rust:
	// State: 17837831107584858481680424076579958046146299570891807727648586105247099570614
	// n_rounds: 100
	// Hash result: 14673626171638495978756045577765290929933355275575684631602375551132464329132

	state, _ := new(big.Int).SetString("17837831107584858481680424076579958046146299570891807727648586105247099570614", 10)
	nRounds := 100
	expectedHash, _ := new(big.Int).SetString("14673626171638495978756045577765290929933355275575684631602375551132464329132", 10)

	assignment := &TestPoseidonHashCircuit{
		State:        state,
		NRounds:      nRounds,
		ExpectedHash: expectedHash,
	}

	var circuit TestPoseidonHashCircuit
	_, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}

	err = test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Poseidon hash mismatch: %v", err)
	}

	t.Log("✓ Poseidon hash matches Rust!")
}

func TestTruncate128ReverseWithSumcheckHash(t *testing.T) {
	// From Rust (with correct n_rounds = 98):
	// Hash result: 19654309276864454743090868671061718225298090476198580618438328100134481672469
	// Hash bytes (LE first 16): [21, 185, 225, 133, 16, 94, 8, 82, 218, 150, 33, 207, 87, 214, 243, 147]
	// r_sumcheck[0] = 6546053814691896674008619906195315247069748678507320726103401577775255757342

	hashResult, _ := new(big.Int).SetString("19654309276864454743090868671061718225298090476198580618438328100134481672469", 10)
	expectedChallenge, _ := new(big.Int).SetString("6546053814691896674008619906195315247069748678507320726103401577775255757342", 10)

	t.Logf("Hash result: %s", hashResult.String())
	t.Logf("Expected challenge: %s", expectedChallenge.String())

	assignment := &TestTruncate128ReverseCircuit{
		Input:    hashResult,
		Expected: expectedChallenge,
	}

	var circuit TestTruncate128ReverseCircuit
	_, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, &circuit)
	if err != nil {
		t.Fatalf("Failed to compile: %v", err)
	}

	err = test.IsSolved(&circuit, assignment, ecc.BN254.ScalarField())
	if err != nil {
		t.Fatalf("Truncate128Reverse mismatch: %v", err)
	}

	t.Log("✓ Truncate128Reverse matches Rust!")
}
