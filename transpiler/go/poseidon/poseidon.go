// Package poseidon implements the Poseidon hash function with circom-compatible parameters
// for BN254. This is adapted from succinctlabs/gnark-plonky2-verifier/poseidon but without
// the Goldilocks dependency.
//
// Parameters: t=4 (width 4), 8 full rounds, 56 partial rounds, S-box x^5
// Constants match light-poseidon circom.
package poseidon

import (
	"math/big"

	"github.com/consensys/gnark/frontend"
)

// pow2 contains precomputed powers of 2 for bit recomposition.
var pow2 [256]*big.Int

func init() {
	for i := 0; i < 256; i++ {
		pow2[i] = new(big.Int).Lsh(big.NewInt(1), uint(i))
	}
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

// ByteReverse performs byte-reversal of a field element using pure constraints.
// Matches Rust: serialize(LE) -> reverse bytes -> from_le_bytes_mod_order.
//
// Decomposes x into 254 bits, then recomposes with reversed byte order.
// Byte i of x (LE) moves to position (31-i). In bits: bit at position
// 8*i+j gets weight 2^(8*(31-i)+j). Recomposition is free in R1CS
// (linear combinations).
func ByteReverse(api frontend.API, x frontend.Variable) frontend.Variable {
	bits := api.ToBinary(x, 254)
	result := frontend.Variable(0)
	for i := 0; i < 32; i++ {
		for j := 0; j < 8; j++ {
			srcBit := i*8 + j
			if srcBit >= 254 {
				continue
			}
			dstPos := (31-i)*8 + j
			result = api.Add(result, api.Mul(bits[srcBit], pow2[dstPos]))
		}
	}
	return result
}

// Truncate128Reverse truncates to 128 bits with Montgomery de-conversion.
// Matches Rust MontU128Challenge::into() for ark_bn254::Fr:
//   - Take low 128 bits of x (bits 0..127)
//   - Apply 125-bit mask (clear bits 125, 126, 127)
//   - Place masked value at Montgomery position: value * 2^128
//   - Multiply by R^-1 mod p
//
// In bits: each bit_i (i=0..124) gets weight 2^(128+i), then * R^-1.
func Truncate128Reverse(api frontend.API, x frontend.Variable) frontend.Variable {
	bits := api.ToBinary(x, 254)
	// Take bits 0..124 (125-bit mask), place at offset 128
	shifted := frontend.Variable(0)
	for i := 0; i < 125; i++ {
		shifted = api.Add(shifted, api.Mul(bits[i], pow2[128+i]))
	}
	// Montgomery de-conversion: multiply by R^-1 mod p
	return api.Mul(shifted, bn254RInv)
}

// bn254RInv is R^-1 mod p for BN254 Fr Montgomery arithmetic.
// R = 2^256 mod p, so R^-1 is the modular inverse of R.
// Used by Truncate128Reverse for Montgomery de-conversion.
var bn254RInv = func() *big.Int {
	rInv, _ := new(big.Int).SetString("9915499612839321149637521777990102151350674507940716049588462388200839649614", 10)
	return rInv
}()

// Truncate128 truncates to 128 bits WITHOUT shifting using pure constraints.
// Matches Rust challenge_scalar_128_bits: take low 16 LE bytes, reverse, interpret as LE.
//
// Decomposes x into 254 bits, takes bits 0..127, recomposes with reversed byte order.
// Bit at position 8*i+j (i=0..15) gets weight 2^(8*(15-i)+j).
func Truncate128(api frontend.API, x frontend.Variable) frontend.Variable {
	bits := api.ToBinary(x, 254)
	result := frontend.Variable(0)
	for i := 0; i < 16; i++ {
		for j := 0; j < 8; j++ {
			srcBit := i*8 + j
			dstPos := (15-i)*8 + j
			result = api.Add(result, api.Mul(bits[srcBit], pow2[dstPos]))
		}
	}
	return result
}

// AppendU64Transform computes bswap64(x) * 2^192 using pure constraints.
// Matches Rust PoseidonTranscript::append_u64: pack x.to_be_bytes() into
// bytes 24-31 of a 32-byte array, interpret as LE field element.
//
// Byte i of x (LE, i=0..7) goes to packed position (31-i).
// Bit at position 8*i+j gets weight 2^(8*(31-i)+j).
func AppendU64Transform(api frontend.API, x frontend.Variable) frontend.Variable {
	bits := api.ToBinary(x, 64)
	result := frontend.Variable(0)
	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			srcBit := i*8 + j
			dstPos := (31-i)*8 + j
			result = api.Add(result, api.Mul(bits[srcBit], pow2[dstPos]))
		}
	}
	return result
}
