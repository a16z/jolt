// Debug circuit: replicate stage1_circuit.go logic with concrete values
// to find where Go and Rust diverge.
//
// This program computes the same values as stage1_circuit.go but with
// concrete fr.Element values instead of frontend.Variable. This lets us
// print intermediate values and compare with Rust.
//
// IMPORTANT: This must produce the same final_check value as the circuit.
// If it doesn't, we haven't replicated the circuit correctly.
//
// Usage: go run ./cmd/debug_circuit
package main

import (
	"encoding/json"
	"fmt"
	"math/big"
	"os"

	"github.com/consensys/gnark-crypto/ecc/bn254/fr"
)

// Witness values from stage1_circuit_test.go
// We'll load these from JSON to match the test

type WitnessData struct {
	Commitments        [41][12]string
	UniSkipCoeffs      [28]string
	SumcheckRoundPolys [11][3]string
	R1csInputs         [36]string
}

// PoseidonHash computes poseidon([0, in1, in2, in3])[0]
// This matches the circuit's poseidon.Hash(api, in1, in2, in3)
func PoseidonHash(in1, in2, in3 *fr.Element) *fr.Element {
	// For now, use a placeholder that matches the circuit structure
	// We need to use the same Poseidon implementation

	// Import the actual poseidon from gnark-crypto
	// gnark-crypto/ecc/bn254/fr/poseidon uses different API
	// Let's just do the computation directly

	// Actually, let's use a test-based approach:
	// We'll compute using the hint functions directly
	panic("TODO: implement Poseidon hash")
}

func main() {
	fmt.Println("=== Debug Circuit (Go) ===")
	fmt.Println()

	// Load witness values from the test file
	// For now, hardcode the key values we need to verify

	// From stage1_circuit.go line 610:
	// The innermost call is: poseidon.Hash(api, 1953263434, 0, 0)
	// "Jolt" = 0x4a6f6c74 as LE = 1953263434

	labelField := new(fr.Element)
	labelField.SetUint64(1953263434)

	fmt.Printf("Label 'Jolt' as field: %s\n", labelField.String())

	// We need to compute poseidon(label, 0, 0) to get initial state
	// But gnark-crypto poseidon has different API than our circuit

	// Let's check what the circuit constants are
	fmt.Println("\n=== Circuit Constants ===")

	// From cse_0 computation, the preamble values are:
	// poseidon.AppendU64Transform(api, 4096)  - max_input_size
	// poseidon.AppendU64Transform(api, 4096)  - max_output_size
	// poseidon.AppendU64Transform(api, 32768) - memory_size
	// 50 - inputs_len (hardcoded, not transformed)
	// 201625236193 - this is the inputs bytes hashed (constant)
	// poseidon.AppendU64Transform(api, 0) - outputs_len
	// poseidon.AppendU64Transform(api, 8192) - panic? No, this is ram_k I think
	// poseidon.AppendU64Transform(api, 1024) - trace_length

	// Let me compute AppendU64Transform for these values
	fmt.Println("\nAppendU64Transform values:")
	printAppendU64Transform(4096, "max_input_size")
	printAppendU64Transform(4096, "max_output_size")
	printAppendU64Transform(32768, "memory_size")
	printAppendU64Transform(0, "outputs_len")
	printAppendU64Transform(8192, "ram_k")
	printAppendU64Transform(1024, "trace_length")

	// The big constant 693065686773592458709161276463075796193455407009757267193429
	// appears in cse_12 - this is "UncompressedUniPoly_begin" encoded
	fmt.Println("\n=== Message Constants ===")
	uniBegin := new(big.Int)
	uniBegin.SetString("693065686773592458709161276463075796193455407009757267193429", 10)
	fmt.Printf("UncompressedUniPoly_begin: %s\n", uniBegin.String())

	// And 9619401173246373414507010453289387209824226095986339413
	// is "UncompressedUniPoly_end" encoded
	uniEnd := new(big.Int)
	uniEnd.SetString("9619401173246373414507010453289387209824226095986339413", 10)
	fmt.Printf("UncompressedUniPoly_end: %s\n", uniEnd.String())

	// Let's verify these are correct encodings
	fmt.Println("\n=== Verify Message Encodings ===")
	verifyMessageEncoding("UncompressedUniPoly_begin")
	verifyMessageEncoding("UncompressedUniPoly_end")
	verifyMessageEncoding("UniPoly_begin")
	verifyMessageEncoding("UniPoly_end")
}

// appendU64TransformHint computes the append_u64 field element transformation.
// This is copied from poseidon/poseidon.go
func appendU64Transform(x uint64) *fr.Element {
	// Pack into 32-byte array with BE padding (like Rust does)
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

func printAppendU64Transform(x uint64, name string) {
	result := appendU64Transform(x)
	fmt.Printf("  %s (%d): %s\n", name, x, result.String())
}

func verifyMessageEncoding(msg string) {
	// Rust does: pad to 32 bytes, interpret as LE field element
	padded := make([]byte, 32)
	copy(padded, []byte(msg))

	// Convert LE to big.Int (reverse to BE)
	be := make([]byte, 32)
	for i := 0; i < 32; i++ {
		be[i] = padded[31-i]
	}
	bigVal := new(big.Int).SetBytes(be)

	fmt.Printf("  '%s': %s\n", msg, bigVal.String())
}

// Load witness from JSON
func loadWitness() (*WitnessData, error) {
	data, err := os.ReadFile("../../data/fib_stage1_data.json")
	if err != nil {
		return nil, err
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}

	// Parse what we need
	// For now just return nil, we'll implement later
	return nil, nil
}
