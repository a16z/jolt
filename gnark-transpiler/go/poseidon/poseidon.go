// Package poseidon implements the Poseidon hash function with circom-compatible parameters
// for BN254. This is adapted from succinctlabs/gnark-plonky2-verifier/poseidon but without
// the Goldilocks dependency.
//
// Parameters: t=4 (width 4), 8 full rounds, 56 partial rounds, S-box x^5
// Constants match light-poseidon circom.
package poseidon

import (
	"math/big"

	"github.com/consensys/gnark/constraint/solver"
	"github.com/consensys/gnark/frontend"
)

func init() {
	solver.RegisterHint(byteReverseHint)
	solver.RegisterHint(truncate128ReverseHint)
	solver.RegisterHint(truncate128Hint)
	solver.RegisterHint(appendU64TransformHint)
}

const BN254_FULL_ROUNDS int = 8
const BN254_PARTIAL_ROUNDS int = 56
const BN254_SPONGE_WIDTH int = 4

type BN254State = [BN254_SPONGE_WIDTH]frontend.Variable

type BN254Chip struct {
	api frontend.API
}

func NewBN254Chip(api frontend.API) *BN254Chip {
	return &BN254Chip{api: api}
}

// Hash computes Poseidon hash of 3 field elements.
// API: poseidon.Hash(api, state, n_rounds, data)
func Hash(api frontend.API, in1, in2, in3 frontend.Variable) frontend.Variable {
	chip := NewBN254Chip(api)

	// Initialize state: [0, in1, in2, in3]
	state := BN254State{
		frontend.Variable(0),
		in1,
		in2,
		in3,
	}

	result := chip.Poseidon(state)
	return result[0]
}

func (c *BN254Chip) Poseidon(state BN254State) BN254State {
	state = c.ark(state, 0)
	state = c.fullRounds(state, true)
	state = c.partialRounds(state)
	state = c.fullRounds(state, false)
	return state
}

func (c *BN254Chip) fullRounds(state BN254State, isFirst bool) BN254State {
	for i := 0; i < BN254_FULL_ROUNDS/2-1; i++ {
		state = c.exp5state(state)
		if isFirst {
			state = c.ark(state, (i+1)*BN254_SPONGE_WIDTH)
		} else {
			state = c.ark(state, (BN254_FULL_ROUNDS/2+1)*BN254_SPONGE_WIDTH+BN254_PARTIAL_ROUNDS+i*BN254_SPONGE_WIDTH)
		}
		state = c.mix(state, mMatrix)
	}

	state = c.exp5state(state)
	if isFirst {
		state = c.ark(state, (BN254_FULL_ROUNDS/2)*BN254_SPONGE_WIDTH)
		state = c.mix(state, pMatrix)
	} else {
		state = c.mix(state, mMatrix)
	}

	return state
}

func (c *BN254Chip) partialRounds(state BN254State) BN254State {
	for i := 0; i < BN254_PARTIAL_ROUNDS; i++ {
		state[0] = c.exp5(state[0])
		state[0] = c.api.Add(state[0], cConstants[(BN254_FULL_ROUNDS/2+1)*BN254_SPONGE_WIDTH+i])

		newState0 := frontend.Variable(0)
		for j := 0; j < BN254_SPONGE_WIDTH; j++ {
			newState0 = c.api.MulAcc(newState0, sConstants[(BN254_SPONGE_WIDTH*2-1)*i+j], state[j])
		}

		for k := 1; k < BN254_SPONGE_WIDTH; k++ {
			state[k] = c.api.MulAcc(state[k], state[0], sConstants[(BN254_SPONGE_WIDTH*2-1)*i+BN254_SPONGE_WIDTH+k-1])
		}
		state[0] = newState0
	}

	return state
}

func (c *BN254Chip) ark(state BN254State, it int) BN254State {
	var result BN254State

	for i := 0; i < len(state); i++ {
		result[i] = c.api.Add(state[i], cConstants[it+i])
	}

	return result
}

func (c *BN254Chip) exp5(x frontend.Variable) frontend.Variable {
	x2 := c.api.Mul(x, x)
	x4 := c.api.Mul(x2, x2)
	return c.api.Mul(x4, x)
}

func (c *BN254Chip) exp5state(state BN254State) BN254State {
	for i := 0; i < BN254_SPONGE_WIDTH; i++ {
		state[i] = c.exp5(state[i])
	}
	return state
}

func (c *BN254Chip) mix(state_ BN254State, constantMatrix [][]*big.Int) BN254State {
	var result BN254State

	for i := 0; i < BN254_SPONGE_WIDTH; i++ {
		result[i] = frontend.Variable(0)
	}

	for i := 0; i < BN254_SPONGE_WIDTH; i++ {
		for j := 0; j < BN254_SPONGE_WIDTH; j++ {
			result[i] = c.api.MulAcc(result[i], constantMatrix[j][i], state_[j])
		}
	}

	return result
}

// ByteReverse performs byte-reversal of a field element.
// This matches Rust PoseidonTranscript::append_scalar behavior:
//
//	serialize(LE) -> reverse bytes -> from_le_bytes_mod_order
//
// For a 256-bit field element stored as 32 bytes [b0, b1, ..., b31] (LE),
// the byte-reversed value interprets [b31, b30, ..., b0] as LE.
func ByteReverse(api frontend.API, x frontend.Variable) frontend.Variable {
	result, err := api.Compiler().NewHint(byteReverseHint, 1, x)
	if err != nil {
		panic(err)
	}
	return result[0]
}

// byteReverseHint computes byte-reverse of a field element.
// Matches Rust: serialize_uncompressed(LE) -> reverse -> from_le_bytes_mod_order
func byteReverseHint(_ *big.Int, inputs []*big.Int, outputs []*big.Int) error {
	if len(inputs) != 1 || len(outputs) != 1 {
		return nil
	}

	// Convert to 32-byte LE representation (like Rust serialize_uncompressed)
	// big.Int stores in big-endian internally, so we need to convert
	inputBytes := inputs[0].Bytes() // This is big-endian

	// Create 32-byte LE array from the BE bytes
	le := make([]byte, 32)
	for i := 0; i < len(inputBytes) && i < 32; i++ {
		le[i] = inputBytes[len(inputBytes)-1-i]
	}

	// Reverse the LE bytes (like Rust buf.reverse())
	reversed := make([]byte, 32)
	for i := 0; i < 32; i++ {
		reversed[i] = le[31-i]
	}

	// Interpret reversed bytes as LE (like Rust from_le_bytes_mod_order)
	// Convert LE to big.Int
	be := make([]byte, 32)
	for i := 0; i < 32; i++ {
		be[i] = reversed[31-i]
	}
	outputs[0] = new(big.Int).SetBytes(be)
	return nil
}

// Truncate128Reverse truncates to 128 bits and byte-reverses.
// This matches Rust PoseidonTranscript::challenge_scalar_128_bits behavior:
//
//	take low 16 bytes (LE) -> reverse -> from_bytes
func Truncate128Reverse(api frontend.API, x frontend.Variable) frontend.Variable {
	result, err := api.Compiler().NewHint(truncate128ReverseHint, 1, x)
	if err != nil {
		panic(err)
	}
	return result[0]
}

// twoPow128 is 2^128 as a big.Int constant
var twoPow128 = func() *big.Int {
	result := new(big.Int).Lsh(big.NewInt(1), 128)
	return result
}()

// twoPow192 is 2^192 as a big.Int constant
var twoPow192Mont = func() *big.Int {
	result := new(big.Int).Lsh(big.NewInt(1), 192)
	return result
}()

// bn254FrModulus is the BN254 Fr field modulus
var bn254FrModulus = func() *big.Int {
	modulus, _ := new(big.Int).SetString("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10)
	return modulus
}()

// bn254RInv is R^-1 mod p for BN254 Fr Montgomery arithmetic
// R = 2^256 mod p, so R^-1 is the modular inverse of R
var bn254RInv = func() *big.Int {
	rInv, _ := new(big.Int).SetString("9915499612839321149637521777990102151350674507940716049588462388200839649614", 10)
	return rInv
}()

// truncate128ReverseHint computes the MontU128Challenge to Fr conversion.
// This matches Rust's MontU128Challenge::into() for ark_bn254::Fr.
//
// Rust MontU128Challenge behavior:
// 1. challenge_u128() gets 16 bytes from hash, reverses, interprets as BE u128
// 2. MontU128Challenge::new(value) applies 125-bit mask and stores [0, 0, low, high]
// 3. Into<Fr> calls Fr::from_bigint_unchecked(BigInt::new([0, 0, low, high]))
// 4. from_bigint_unchecked treats input as Montgomery form and multiplies by R^-1
//
// So the final value is: (low * 2^128 + high * 2^192) * R^-1 mod p
func truncate128ReverseHint(_ *big.Int, inputs []*big.Int, outputs []*big.Int) error {
	if len(inputs) != 1 || len(outputs) != 1 {
		return nil
	}

	// Convert to 32-byte LE representation (like Rust serialize_uncompressed)
	inputBytes := inputs[0].Bytes() // big-endian from big.Int
	le := make([]byte, 32)
	for i := 0; i < len(inputBytes) && i < 32; i++ {
		le[i] = inputBytes[len(inputBytes)-1-i]
	}

	// Take first 16 bytes (low 128 bits in LE format)
	le16 := le[:16]

	// Reverse 16 bytes (like Rust buf.reverse())
	for i := 0; i < 8; i++ {
		le16[i], le16[15-i] = le16[15-i], le16[i]
	}

	// Interpret as BE to get u128 value (like Rust u128::from_be_bytes)
	value128 := new(big.Int).SetBytes(le16)

	// Apply 125-bit mask for MontU128Challenge
	mask125 := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 125), big.NewInt(1))
	valueMasked := new(big.Int).And(value128, mask125)

	// Compute BigInt representation [0, 0, low, high]
	// This equals: low * 2^128 + high * 2^192
	low := new(big.Int).And(valueMasked, new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 64), big.NewInt(1)))
	high := new(big.Int).Rsh(valueMasked, 64)

	bigintValue := new(big.Int).Add(
		new(big.Int).Mul(low, twoPow128),
		new(big.Int).Mul(high, twoPow192Mont),
	)

	// Multiply by R^-1 mod p (what from_bigint_unchecked does)
	result := new(big.Int).Mul(bigintValue, bn254RInv)
	result.Mod(result, bn254FrModulus)

	outputs[0] = result
	return nil
}

// Truncate128 truncates to 128 bits WITHOUT shifting.
// This matches Rust PoseidonTranscript::challenge_scalar behavior which returns Fr directly.
//
// Used for: r0 (univariate-skip), batching_coeff, sumcheck challenges
//
// Transforms: take low 16 bytes (LE) -> interpret as u128 -> return as Fr (no mask, no shift)
func Truncate128(api frontend.API, x frontend.Variable) frontend.Variable {
	result, err := api.Compiler().NewHint(truncate128Hint, 1, x)
	if err != nil {
		panic(err)
	}
	return result[0]
}

// truncate128Hint computes truncate-to-128-bits WITHOUT shifting.
// This matches Rust challenge_scalar behavior which truncates to 128 bits and returns Fr.
//
// Rust challenge_scalar_128_bits does:
// 1. challenge_bytes(&mut buf) - fills 16 bytes from hash
// 2. buf.reverse() - reverse the 16 bytes
// 3. JF::from_bytes(&buf) = Fr::from_le_bytes_mod_order(&buf) - interpret as LE
//
// Note: Unlike challenge_scalar_optimized (MontU128Challenge), this does NOT:
// - Apply 125-bit mask
// - Shift by 2^128
func truncate128Hint(_ *big.Int, inputs []*big.Int, outputs []*big.Int) error {
	if len(inputs) != 1 || len(outputs) != 1 {
		return nil
	}

	// Convert to 32-byte LE representation (like Rust serialize_uncompressed)
	inputBytes := inputs[0].Bytes() // big-endian from big.Int
	le := make([]byte, 32)
	for i := 0; i < len(inputBytes) && i < 32; i++ {
		le[i] = inputBytes[len(inputBytes)-1-i]
	}

	// Step 1: Take first 16 bytes (low 128 bits in LE format)
	le16 := make([]byte, 16)
	copy(le16, le[:16])

	// Step 2: Reverse the 16 bytes (like Rust buf.reverse())
	for i := 0; i < 8; i++ {
		le16[i], le16[15-i] = le16[15-i], le16[i]
	}

	// Step 3: Interpret the reversed bytes as LE (like Rust from_le_bytes_mod_order)
	// To convert LE to big.Int, we reverse to get BE
	be := make([]byte, 16)
	for i := 0; i < 16; i++ {
		be[i] = le16[15-i]
	}
	outputs[0] = new(big.Int).SetBytes(be)
	return nil
}

// twoPow192 is 2^192 as a big.Int constant
// 2^192 = 6277101735386680763835789423207666416102355444464034512896
var twoPow192 = func() *big.Int {
	result := new(big.Int).Lsh(big.NewInt(1), 192)
	return result
}()

// AppendU64Transform computes the field element for append_u64(x).
//
// PoseidonTranscript::append_u64 does:
// 1. Pack u64 x into bytes 24-31 of a 32-byte array using x.to_be_bytes()
// 2. Interpret the 32 bytes as a little-endian field element
//
// For x with BE bytes [b7,b6,b5,b4,b3,b2,b1,b0] (where b7 is MSB):
// - packed[24..32] = [b7,b6,b5,b4,b3,b2,b1,b0]
// - LE interpretation = b7*2^192 + b6*2^200 + ... + b0*2^248
//
// This is equivalent to: bswap64(x) * 2^192
// where bswap64 reverses the byte order of the u64.
//
// NOTE: This uses a hint because the transformation involves byte manipulation
// that can't be expressed efficiently with field arithmetic.
func AppendU64Transform(api frontend.API, x frontend.Variable) frontend.Variable {
	result, err := api.Compiler().NewHint(appendU64TransformHint, 1, x)
	if err != nil {
		panic(err)
	}
	return result[0]
}

// appendU64TransformHint computes the append_u64 field element transformation.
func appendU64TransformHint(_ *big.Int, inputs []*big.Int, outputs []*big.Int) error {
	if len(inputs) != 1 || len(outputs) != 1 {
		return nil
	}

	x := inputs[0].Uint64()

	// Pack into 32-byte array with BE padding (like Rust does)
	packed := make([]byte, 32)
	// x.to_be_bytes() puts MSB first
	packed[24] = byte(x >> 56)
	packed[25] = byte(x >> 48)
	packed[26] = byte(x >> 40)
	packed[27] = byte(x >> 32)
	packed[28] = byte(x >> 24)
	packed[29] = byte(x >> 16)
	packed[30] = byte(x >> 8)
	packed[31] = byte(x)

	// Interpret as LE: convert to big.Int by reversing to BE
	be := make([]byte, 32)
	for i := 0; i < 32; i++ {
		be[i] = packed[31-i]
	}
	outputs[0] = new(big.Int).SetBytes(be)
	return nil
}

// MulTwoPow192 multiplies a field element by 2^192.
// DEPRECATED: Use AppendU64Transform for append_u64 operations.
// This function is kept for backwards compatibility but is incorrect for append_u64.
func MulTwoPow192(api frontend.API, x frontend.Variable) frontend.Variable {
	return api.Mul(x, twoPow192)
}
