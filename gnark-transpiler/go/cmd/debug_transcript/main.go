// Debug transcript: compute Poseidon hashes step by step with concrete values
// to compare against Rust transcript.
//
// Usage: go run ./cmd/debug_transcript
package main

import (
	"encoding/json"
	"fmt"
	"math/big"
	"os"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
	"github.com/consensys/gnark-crypto/ecc/bn254/fr/poseidon"
)

// PoseidonHash computes poseidon(in1, in2, in3) -> Fr
// Matches the circuit's poseidon.Hash(api, in1, in2, in3)
func PoseidonHash(in1, in2, in3 *fr.Element) *fr.Element {
	// State: [0, in1, in2, in3]
	state := make([]*fr.Element, 4)
	state[0] = new(fr.Element).SetZero()
	state[1] = new(fr.Element).Set(in1)
	state[2] = new(fr.Element).Set(in2)
	state[3] = new(fr.Element).Set(in3)

	// Use gnark-crypto's Poseidon permutation
	result := poseidon.Permutation(state)
	return result[0]
}

// ByteReverse performs the same transformation as the Go hint
func ByteReverse(x *fr.Element) *fr.Element {
	// Get 32-byte LE representation
	var le [32]byte
	x.LittleEndian(le[:])

	// Reverse bytes
	var reversed [32]byte
	for i := 0; i < 32; i++ {
		reversed[i] = le[31-i]
	}

	// Interpret reversed bytes as LE field element
	result := new(fr.Element)
	result.SetBytesCanonical(reversed[:]) // This expects big-endian

	// Actually we need to interpret reversed as LE...
	// Let's do it properly: reverse to BE for SetBytes
	var be [32]byte
	for i := 0; i < 32; i++ {
		be[i] = reversed[31-i]
	}
	result.SetBytes(be[:])
	return result
}

// Truncate128Reverse performs the same transformation as the Go hint
func Truncate128Reverse(x *fr.Element) *fr.Element {
	// Get 32-byte LE representation
	var le [32]byte
	x.LittleEndian(le[:])

	// Take first 16 bytes (low 128 bits)
	le16 := le[:16]

	// Interpret as u128 in LE
	// Convert to big.Int (need BE for SetBytes)
	be := make([]byte, 16)
	for i := 0; i < 16; i++ {
		be[i] = le16[15-i]
	}
	value128 := new(big.Int).SetBytes(be)

	// Mask to 125 bits
	mask125 := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 125), big.NewInt(1))
	valueMasked := new(big.Int).And(value128, mask125)

	// Shift left by 128 bits
	twoPow128 := new(big.Int).Lsh(big.NewInt(1), 128)
	shifted := new(big.Int).Mul(valueMasked, twoPow128)

	// Convert back to Fr
	result := new(fr.Element)
	result.SetBigInt(shifted)
	return result
}

// AppendU64Transform computes the field element for append_u64(x)
func AppendU64Transform(x uint64) *fr.Element {
	// Pack into 32-byte array with BE padding
	packed := make([]byte, 32)
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
	bigVal := new(big.Int).SetBytes(be)

	result := new(fr.Element)
	result.SetBigInt(bigVal)
	return result
}

// Extracted data structure
type ExtractedData struct {
	Preamble struct {
		MaxInputSize  uint64 `json:"max_input_size"`
		MaxOutputSize uint64 `json:"max_output_size"`
		MemorySize    uint64 `json:"memory_size"`
		Panic         bool   `json:"panic"`
		RamK          uint64 `json:"ram_k"`
		TraceLength   uint64 `json:"trace_length"`
		Inputs        []byte `json:"inputs"`
		Outputs       []byte `json:"outputs"`
	} `json:"preamble"`
	Commitments         [][]byte   `json:"commitments"`
	UniSkipPolyCoeffs   []string   `json:"uni_skip_poly_coeffs"`
	SumcheckRoundPolys  [][]string `json:"sumcheck_round_polys"`
	ExpectedFinalClaim  string     `json:"expected_final_claim"`
}

func main() {
	// Load extracted data
	dataPath := "../../data/fib_stage1_data.json"
	dataBytes, err := os.ReadFile(dataPath)
	if err != nil {
		fmt.Printf("Error reading data: %v\n", err)
		return
	}

	var data ExtractedData
	if err := json.Unmarshal(dataBytes, &data); err != nil {
		fmt.Printf("Error parsing JSON: %v\n", err)
		return
	}

	fmt.Println("=== Debug Transcript (Go) ===")
	fmt.Println()

	// Step 1: Initialize transcript with "Jolt" label
	// "Jolt" = [0x4a, 0x6f, 0x6c, 0x74] -> as LE u64 = 0x746c6f4a = 1953263434
	labelField := new(fr.Element)
	labelField.SetUint64(1953263434)

	zero := new(fr.Element).SetZero()
	state := PoseidonHash(labelField, zero, zero)
	nRounds := uint64(0)

	fmt.Printf("After init (poseidon('Jolt', 0, 0)):\n")
	fmt.Printf("  state = %s\n", state.String())
	fmt.Println()

	// Step 2: Preamble - append_u64 for each value
	preambleValues := []uint64{
		data.Preamble.MaxInputSize,
		data.Preamble.MaxOutputSize,
		data.Preamble.MemorySize,
	}
	preambleNames := []string{"max_input_size", "max_output_size", "memory_size"}

	for i, val := range preambleValues {
		transformed := AppendU64Transform(val)
		round := new(fr.Element).SetUint64(nRounds)
		state = PoseidonHash(state, round, transformed)
		nRounds++
		fmt.Printf("After preamble %s (%d), n_rounds=%d:\n", preambleNames[i], val, nRounds-1)
		fmt.Printf("  transformed = %s\n", transformed.String())
		fmt.Printf("  state = %s\n", state.String())
	}

	// inputs_len (len of inputs array as u64)
	inputsLen := uint64(len(data.Preamble.Inputs))
	transformed := AppendU64Transform(inputsLen)
	round := new(fr.Element).SetUint64(nRounds)
	state = PoseidonHash(state, round, transformed)
	nRounds++
	fmt.Printf("After inputs_len (%d), n_rounds=%d:\n", inputsLen, nRounds-1)
	fmt.Printf("  state = %s\n", state.String())

	// inputs bytes (each byte as append_u64)
	for i, b := range data.Preamble.Inputs {
		transformed := AppendU64Transform(uint64(b))
		round := new(fr.Element).SetUint64(nRounds)
		state = PoseidonHash(state, round, transformed)
		nRounds++
		if i < 3 {
			fmt.Printf("After inputs[%d] (%d), n_rounds=%d:\n", i, b, nRounds-1)
			fmt.Printf("  state = %s\n", state.String())
		}
	}
	fmt.Printf("... (skipping %d more input bytes)\n", len(data.Preamble.Inputs)-3)

	// outputs_len
	outputsLen := uint64(len(data.Preamble.Outputs))
	transformed = AppendU64Transform(outputsLen)
	round = new(fr.Element).SetUint64(nRounds)
	state = PoseidonHash(state, round, transformed)
	nRounds++
	fmt.Printf("After outputs_len (%d), n_rounds=%d:\n", outputsLen, nRounds-1)
	fmt.Printf("  state = %s\n", state.String())

	// outputs bytes
	for i, b := range data.Preamble.Outputs {
		transformed := AppendU64Transform(uint64(b))
		round := new(fr.Element).SetUint64(nRounds)
		state = PoseidonHash(state, round, transformed)
		nRounds++
		if i < 3 {
			fmt.Printf("After outputs[%d] (%d), n_rounds=%d:\n", i, b, nRounds-1)
			fmt.Printf("  state = %s\n", state.String())
		}
	}
	fmt.Printf("... (skipping %d more output bytes)\n", len(data.Preamble.Outputs)-3)

	// panic
	panicVal := uint64(0)
	if data.Preamble.Panic {
		panicVal = 1
	}
	transformed = AppendU64Transform(panicVal)
	round = new(fr.Element).SetUint64(nRounds)
	state = PoseidonHash(state, round, transformed)
	nRounds++
	fmt.Printf("After panic (%d), n_rounds=%d:\n", panicVal, nRounds-1)
	fmt.Printf("  state = %s\n", state.String())

	// ram_k
	transformed = AppendU64Transform(data.Preamble.RamK)
	round = new(fr.Element).SetUint64(nRounds)
	state = PoseidonHash(state, round, transformed)
	nRounds++
	fmt.Printf("After ram_k (%d), n_rounds=%d:\n", data.Preamble.RamK, nRounds-1)
	fmt.Printf("  state = %s\n", state.String())

	// trace_length
	transformed = AppendU64Transform(data.Preamble.TraceLength)
	round = new(fr.Element).SetUint64(nRounds)
	state = PoseidonHash(state, round, transformed)
	nRounds++
	fmt.Printf("After trace_length (%d), n_rounds=%d:\n", data.Preamble.TraceLength, nRounds-1)
	fmt.Printf("  state = %s\n", state.String())
	fmt.Println()

	// Step 3: Commitments
	// Each commitment is 384 bytes (already reversed in JSON)
	// We chunk into 12 x 32 bytes and hash with chaining
	fmt.Printf("=== COMMITMENTS (n_rounds before = %d) ===\n", nRounds)

	for c := 0; c < len(data.Commitments); c++ {
		commitment := data.Commitments[c]

		// Chunk into 12 pieces of 32 bytes
		round := new(fr.Element).SetUint64(nRounds)

		// First chunk uses n_rounds
		chunk0 := chunkToFr(commitment, 0)
		state = PoseidonHash(state, round, chunk0)

		// Remaining chunks use 0
		for i := 1; i < 12; i++ {
			chunk := chunkToFr(commitment, i)
			state = PoseidonHash(state, zero, chunk)
		}
		nRounds++

		if c < 3 {
			fmt.Printf("After commitment[%d], n_rounds=%d:\n", c, nRounds-1)
			fmt.Printf("  state = %s\n", state.String())
		}
	}
	fmt.Printf("After all %d commitments, n_rounds=%d\n", len(data.Commitments), nRounds)
	fmt.Printf("  state = %s\n", state.String())
	fmt.Println()

	// Derive first challenge (r_spartan)
	round = new(fr.Element).SetUint64(nRounds)
	challengeHash := PoseidonHash(state, round, zero)
	state = challengeHash
	nRounds++
	challenge := Truncate128Reverse(challengeHash)
	fmt.Printf("After r_spartan challenge (n_rounds=%d):\n", nRounds-1)
	fmt.Printf("  challenge_hash = %s\n", challengeHash.String())
	fmt.Printf("  challenge (truncated) = %s\n", challenge.String())
	fmt.Println()

	fmt.Printf("DONE. Final n_rounds = %d\n", nRounds)
}

// chunkToFr extracts 32 bytes from commitment and converts to Fr
func chunkToFr(commitment []byte, chunkIdx int) *fr.Element {
	start := chunkIdx * 32
	end := start + 32
	if end > len(commitment) {
		end = len(commitment)
	}

	chunk := make([]byte, 32)
	if start < len(commitment) {
		copy(chunk, commitment[start:end])
	}

	// Interpret as LE: Fr::from_le_bytes_mod_order
	// Need to convert LE to BE for big.Int
	be := make([]byte, 32)
	for i := 0; i < 32; i++ {
		be[i] = chunk[31-i]
	}
	bigVal := new(big.Int).SetBytes(be)

	// Reduce mod field order
	modulus, _ := new(big.Int).SetString("21888242871839275222246405745257275088548364400416034343698204186575808495617", 10)
	bigVal.Mod(bigVal, modulus)

	result := new(fr.Element)
	result.SetBigInt(bigVal)
	return result
}
