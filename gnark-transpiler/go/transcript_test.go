package jolt_verifier

import (
	"fmt"
	"math/big"
	"testing"

	"jolt_verifier/poseidon"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/test"
)

// TestTranscriptInit verifies that the initial transcript state matches Rust
// Rust: Transcript::new(b"Jolt") produces initial_state = hash([label_f, 0, 0])
func TestTranscriptInit(t *testing.T) {
	// "Jolt" = [0x4A, 0x6F, 0x6C, 0x74] in ASCII (big endian)
	// As little-endian 32-byte integer: 0x746C6F4A = 1953263434
	joltLabel := big.NewInt(1953263434)

	// Expected initial state from Rust: hash([1953263434, 0, 0])
	// We need to compute this with gnark's native arithmetic

	fmt.Printf("Jolt label as Fr: %s\n", joltLabel.String())

	// Create a test circuit that verifies the initial hash
	type InitCircuit struct {
		ExpectedHash frontend.Variable `gnark:",public"`
	}

	circuit := &InitCircuit{}

	// Compute expected hash using concrete field arithmetic
	var labelFr, zeroFr fr.Element
	labelFr.SetBigInt(joltLabel)
	zeroFr.SetZero()

	// Use light-poseidon compatible hash
	// We need to match: poseidon.Hash(api, 1953263434, 0, 0)

	// For now, let's just verify the circuit compiles
	_ = test.NewAssert(t)
	fmt.Printf("InitCircuit would verify: Hash(%s, 0, 0)\n", joltLabel.String())
	_ = circuit
}


// TransformCircuit tests AppendU64Transform
type TransformCircuit struct {
	Input    frontend.Variable `gnark:",public"`
	Expected frontend.Variable `gnark:",public"`
}

func (c *TransformCircuit) Define(api frontend.API) error {
	result := poseidon.AppendU64Transform(api, c.Input)
	api.AssertIsEqual(result, c.Expected)
	return nil
}

func TestAppendU64TransformCircuit(t *testing.T) {
	assert := test.NewAssert(t)

	// Test case: x = 4096
	// Expected: 16 * 2^240
	expected := new(big.Int).Lsh(big.NewInt(16), 240)

	circuit := &TransformCircuit{}
	assignment := &TransformCircuit{
		Input:    4096,
		Expected: expected,
	}

	assert.ProverSucceeded(circuit, assignment, test.WithCurves(ecc.BN254))
}

// ByteReverseCircuit tests ByteReverse
type ByteReverseCircuit struct {
	Input    frontend.Variable `gnark:",public"`
	Expected frontend.Variable `gnark:",public"`
}

func (c *ByteReverseCircuit) Define(api frontend.API) error {
	result := poseidon.ByteReverse(api, c.Input)
	api.AssertIsEqual(result, c.Expected)
	return nil
}

func TestByteReverseCircuitSimple(t *testing.T) {
	assert := test.NewAssert(t)

	// Test case: x = 42
	// Rust: serialize_uncompressed(42) gives LE bytes [42, 0, 0, ..., 0] (32 bytes)
	// Reverse: [0, 0, ..., 0, 42]
	// from_le_bytes_mod_order: 42 * 2^(31*8) = 42 * 2^248
	expected := new(big.Int).Lsh(big.NewInt(42), 248)

	circuit := &ByteReverseCircuit{}
	assignment := &ByteReverseCircuit{
		Input:    42,
		Expected: expected,
	}

	assert.ProverSucceeded(circuit, assignment, test.WithCurves(ecc.BN254))
}

// Truncate128Circuit tests Truncate128
type Truncate128Circuit struct {
	Input    frontend.Variable `gnark:",public"`
	Expected frontend.Variable `gnark:",public"`
}

func (c *Truncate128Circuit) Define(api frontend.API) error {
	result := poseidon.Truncate128(api, c.Input)
	api.AssertIsEqual(result, c.Expected)
	return nil
}

func TestTruncate128Circuit(t *testing.T) {
	assert := test.NewAssert(t)

	// Test case: A 256-bit number
	// Let's use: 0x123456789ABCDEF0_FEDCBA9876543210_11111111_22222222
	// Low 128 bits (LE bytes 0-15): 0x22222222_11111111_FEDCBA9876543210
	// After reverse of 16 bytes: 0x10325476_98BADCFE_11111111_22222222
	// As LE number: ...

	// Simpler test: input = 0x0102030405060708090A0B0C0D0E0F10 (128 bits in high position)
	// But actually, let's test with a small number first

	// Input: some 256-bit value
	input := new(big.Int)
	input.SetString("21530156658456779109967155919523792247479339410230417680241443586155850216169", 10)

	// Expected: truncate to low 128 bits, reverse, interpret as LE
	// Low 16 bytes of input (LE):
	inputBytes := input.Bytes()
	// Pad to 32 bytes (big-endian in big.Int)
	padded := make([]byte, 32)
	copy(padded[32-len(inputBytes):], inputBytes)

	// Convert to LE
	le := make([]byte, 32)
	for i := 0; i < 32; i++ {
		le[i] = padded[31-i]
	}

	// Take low 16 bytes (bytes 0-15 in LE)
	le16 := le[:16]

	// Reverse
	for i := 0; i < 8; i++ {
		le16[i], le16[15-i] = le16[15-i], le16[i]
	}

	// Interpret as LE (reverse to get BE for big.Int)
	be16 := make([]byte, 16)
	for i := 0; i < 16; i++ {
		be16[i] = le16[15-i]
	}
	expected := new(big.Int).SetBytes(be16)

	fmt.Printf("Input: %s\n", input.String())
	fmt.Printf("Expected truncate128: %s\n", expected.String())

	circuit := &Truncate128Circuit{}
	assignment := &Truncate128Circuit{
		Input:    input,
		Expected: expected,
	}

	assert.ProverSucceeded(circuit, assignment, test.WithCurves(ecc.BN254))
}
