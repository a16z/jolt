package poseidon

import (
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

func TestPoseidonConstants(t *testing.T) {
	// Verify first constant matches light-poseidon circom t=4
	// ark[0] = 11633431549750490989983886834189948010834808234699737327785600195936805266405
	expected := new(big.Int)
	expected.SetString("11633431549750490989983886834189948010834808234699737327785600195936805266405", 10)

	if cConstants[0].Cmp(expected) != 0 {
		t.Errorf("cConstants[0] mismatch!\nGot:      %s\nExpected: %s", cConstants[0].String(), expected.String())
	}
}

// =============================================================================
// Tests for Jolt-specific utility functions (lines 143+ in poseidon.go)
// These functions implement byte manipulations matching Rust transcript behavior.
// =============================================================================

// TestByteReverseCircuit tests the ByteReverse function
type TestByteReverseCircuit struct {
	Input    frontend.Variable
	Expected frontend.Variable `gnark:",public"`
}

func (c *TestByteReverseCircuit) Define(api frontend.API) error {
	result := ByteReverse(api, c.Input)
	api.AssertIsEqual(result, c.Expected)
	return nil
}

func TestByteReverse(t *testing.T) {
	// ByteReverse swaps byte order of a 32-byte LE representation.
	// Byte i goes to position (31-i).
	//
	// Input: 0x0102030405060708 (8 bytes in LE: [08,07,06,05,04,03,02,01,00,...,00])
	// After reversal: byte 0 (0x08) -> pos 31, byte 1 (0x07) -> pos 30, ...
	// Result bytes: [00,...,00,01,02,03,04,05,06,07,08] (byte 31=0x08, byte 30=0x07,...)
	// As LE integer: 0x08*2^(31*8) + 0x07*2^(30*8) + ... + 0x01*2^(24*8)
	assert := test.NewAssert(t)
	var circuit TestByteReverseCircuit

	input := new(big.Int)
	input.SetString("0102030405060708", 16)

	// Compute expected by manually reversing bytes
	// Input LE bytes: [0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0, ..., 0]
	// Output LE bytes: [0, ..., 0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
	//                  (positions 24-31)
	expected := new(big.Int)
	expected.Add(expected, new(big.Int).Lsh(big.NewInt(0x08), 31*8))
	expected.Add(expected, new(big.Int).Lsh(big.NewInt(0x07), 30*8))
	expected.Add(expected, new(big.Int).Lsh(big.NewInt(0x06), 29*8))
	expected.Add(expected, new(big.Int).Lsh(big.NewInt(0x05), 28*8))
	expected.Add(expected, new(big.Int).Lsh(big.NewInt(0x04), 27*8))
	expected.Add(expected, new(big.Int).Lsh(big.NewInt(0x03), 26*8))
	expected.Add(expected, new(big.Int).Lsh(big.NewInt(0x02), 25*8))
	expected.Add(expected, new(big.Int).Lsh(big.NewInt(0x01), 24*8))

	assignment := &TestByteReverseCircuit{
		Input:    input,
		Expected: expected,
	}

	assert.ProverSucceeded(&circuit, assignment, test.WithCurves(ecc.BN254))
}

func TestByteReverseMultipleBytes(t *testing.T) {
	// Test with 16-byte value to verify middle bytes are handled correctly
	// Input: 0x0102030405060708090a0b0c0d0e0f10 (16 bytes)
	// LE bytes [0..15]: [0x10,0x0f,0x0e,0x0d,0x0c,0x0b,0x0a,0x09,0x08,0x07,0x06,0x05,0x04,0x03,0x02,0x01]
	// After reversal: byte i -> position (31-i)
	// byte 0 (0x10) -> pos 31, byte 1 (0x0f) -> pos 30, ..., byte 15 (0x01) -> pos 16
	assert := test.NewAssert(t)
	var circuit TestByteReverseCircuit

	input := new(big.Int)
	input.SetString("0102030405060708090a0b0c0d0e0f10", 16)

	// Compute expected
	inputBytes := []byte{0x10, 0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01}
	expected := new(big.Int)
	for i, b := range inputBytes {
		pos := (31 - i) * 8
		expected.Add(expected, new(big.Int).Lsh(big.NewInt(int64(b)), uint(pos)))
	}

	assignment := &TestByteReverseCircuit{
		Input:    input,
		Expected: expected,
	}

	assert.ProverSucceeded(&circuit, assignment, test.WithCurves(ecc.BN254))
}

// TestTruncate128ReverseCircuit tests the Truncate128Reverse function
// This matches Rust's MontU128Challenge::into() for sumcheck challenges.
type TestTruncate128ReverseCircuit struct {
	Input    frontend.Variable
	Expected frontend.Variable `gnark:",public"`
}

func (c *TestTruncate128ReverseCircuit) Define(api frontend.API) error {
	result := Truncate128Reverse(api, c.Input)
	api.AssertIsEqual(result, c.Expected)
	return nil
}

func TestTruncate128Reverse(t *testing.T) {
	// Truncate128Reverse takes low 125 bits, shifts by 128, multiplies by R^-1
	// This matches Jolt's from_bigint_unchecked behavior.
	//
	// For input = 1:
	// - Low 125 bits = 1
	// - Shifted: 1 * 2^128
	// - Result: (1 * 2^128) * R^-1 mod p
	//
	// R^-1 = 9915499612839321149637521777990102151350674507940716049588462388200839649614
	// 2^128 = 340282366920938463463374607431768211456
	// Result = (2^128 * R^-1) mod p
	assert := test.NewAssert(t)
	var circuit TestTruncate128ReverseCircuit

	// Compute expected: (2^128 * R^-1) mod p
	// BN254 Fr modulus
	p := new(big.Int)
	p.SetString("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10)

	rInv := new(big.Int)
	rInv.SetString("9915499612839321149637521777990102151350674507940716049588462388200839649614", 10)

	twoTo128 := new(big.Int).Lsh(big.NewInt(1), 128)

	expected := new(big.Int)
	expected.Mul(twoTo128, rInv)
	expected.Mod(expected, p)

	assignment := &TestTruncate128ReverseCircuit{
		Input:    1,
		Expected: expected,
	}

	assert.ProverSucceeded(&circuit, assignment, test.WithCurves(ecc.BN254))
}

func TestTruncate128ReverseMasking(t *testing.T) {
	// Test that bits 125-127 are masked out (125-bit mask applied)
	// Input with bit 125 set should give same result as input without it
	//
	// Input1 = 2^125 - 1 (all 125 bits set)
	// Input2 = 2^128 - 1 (all 128 bits set, but bits 125-127 should be masked)
	// Both should produce the same output
	assert := test.NewAssert(t)
	var circuit TestTruncate128ReverseCircuit

	p := new(big.Int)
	p.SetString("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10)

	rInv := new(big.Int)
	rInv.SetString("9915499612839321149637521777990102151350674507940716049588462388200839649614", 10)

	// Input with only low 125 bits set
	input125 := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 125), big.NewInt(1))

	// Expected: (input125 * 2^128 * R^-1) mod p
	expected := new(big.Int)
	expected.Lsh(input125, 128)
	expected.Mul(expected, rInv)
	expected.Mod(expected, p)

	// Test with input that has extra bits set (they should be masked)
	input128 := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 128), big.NewInt(1))

	assignment := &TestTruncate128ReverseCircuit{
		Input:    input128,
		Expected: expected, // Should match 125-bit input result
	}

	assert.ProverSucceeded(&circuit, assignment, test.WithCurves(ecc.BN254))
}

// TestTruncate128Circuit tests the Truncate128 function (no Montgomery conversion)
type TestTruncate128Circuit struct {
	Input    frontend.Variable
	Expected frontend.Variable `gnark:",public"`
}

func (c *TestTruncate128Circuit) Define(api frontend.API) error {
	result := Truncate128(api, c.Input)
	api.AssertIsEqual(result, c.Expected)
	return nil
}

func TestTruncate128(t *testing.T) {
	// Truncate128 takes low 128 bits and reverses byte order within those 16 bytes
	// Input: 0x0102030405060708090a0b0c0d0e0f10 (16 bytes)
	// LE bytes: [10, 0f, 0e, 0d, 0c, 0b, 0a, 09, 08, 07, 06, 05, 04, 03, 02, 01]
	// Reversed: [01, 02, 03, 04, 05, 06, 07, 08, 09, 0a, 0b, 0c, 0d, 0e, 0f, 10]
	// As LE:    0x100f0e0d0c0b0a090807060504030201
	assert := test.NewAssert(t)
	var circuit TestTruncate128Circuit

	input := new(big.Int)
	input.SetString("0102030405060708090a0b0c0d0e0f10", 16)

	expected := new(big.Int)
	expected.SetString("100f0e0d0c0b0a090807060504030201", 16)

	assignment := &TestTruncate128Circuit{
		Input:    input,
		Expected: expected,
	}

	assert.ProverSucceeded(&circuit, assignment, test.WithCurves(ecc.BN254))
}

func TestTruncate128IgnoresHighBits(t *testing.T) {
	// Test that Truncate128 ignores bits above 128
	// Input: large value with bits set above 128
	// Output should only depend on low 128 bits
	assert := test.NewAssert(t)
	var circuit TestTruncate128Circuit

	// Low 128 bits: 0x0102030405060708090a0b0c0d0e0f10
	low128 := new(big.Int)
	low128.SetString("0102030405060708090a0b0c0d0e0f10", 16)

	// Add high bits: low128 + (0xdeadbeef * 2^128)
	input := new(big.Int)
	highPart := new(big.Int)
	highPart.SetString("deadbeef", 16)
	highPart.Lsh(highPart, 128)
	input.Add(low128, highPart)

	// Expected is same as if we only had low 128 bits
	expected := new(big.Int)
	expected.SetString("100f0e0d0c0b0a090807060504030201", 16)

	assignment := &TestTruncate128Circuit{
		Input:    input,
		Expected: expected,
	}

	assert.ProverSucceeded(&circuit, assignment, test.WithCurves(ecc.BN254))
}

// TestAppendU64TransformCircuit tests the AppendU64Transform function
type TestAppendU64TransformCircuit struct {
	Input    frontend.Variable
	Expected frontend.Variable `gnark:",public"`
}

func (c *TestAppendU64TransformCircuit) Define(api frontend.API) error {
	result := AppendU64Transform(api, c.Input)
	api.AssertIsEqual(result, c.Expected)
	return nil
}

func TestAppendU64Transform(t *testing.T) {
	// AppendU64Transform computes bswap64(x) * 2^192
	// This matches Rust PoseidonTranscript::append_u64 behavior:
	// pack x.to_be_bytes() into bytes 24-31 of a 32-byte array, interpret as LE.
	//
	// Input: 0x0102030405060708
	// BE bytes: [01, 02, 03, 04, 05, 06, 07, 08]
	// These go to positions 24-31
	// As LE field element: 0x0807060504030201 * 2^192
	assert := test.NewAssert(t)
	var circuit TestAppendU64TransformCircuit

	input := new(big.Int)
	input.SetString("0102030405060708", 16)

	// bswap64(0x0102030405060708) = 0x0807060504030201
	// Result = 0x0807060504030201 * 2^192
	bswapped := new(big.Int)
	bswapped.SetString("0807060504030201", 16)
	expected := new(big.Int).Lsh(bswapped, 192)

	assignment := &TestAppendU64TransformCircuit{
		Input:    input,
		Expected: expected,
	}

	assert.ProverSucceeded(&circuit, assignment, test.WithCurves(ecc.BN254))
}

func TestAppendU64TransformRoundNumber(t *testing.T) {
	// Test with a typical round number value (small integer)
	// This is the common use case in transcript operations
	//
	// Input: 42 (0x2a)
	// BE bytes: [00, 00, 00, 00, 00, 00, 00, 2a]
	// As bswap64: 0x2a00000000000000
	// Result: 0x2a00000000000000 * 2^192
	assert := test.NewAssert(t)
	var circuit TestAppendU64TransformCircuit

	input := big.NewInt(42)

	// bswap64(42) = 42 * 2^56 = 0x2a00000000000000
	bswapped := new(big.Int).Lsh(big.NewInt(42), 56)
	expected := new(big.Int).Lsh(bswapped, 192)

	assignment := &TestAppendU64TransformCircuit{
		Input:    input,
		Expected: expected,
	}

	assert.ProverSucceeded(&circuit, assignment, test.WithCurves(ecc.BN254))
}
