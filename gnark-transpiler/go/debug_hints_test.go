package jolt_verifier

import (
	"fmt"
	"math/big"
	"testing"
)

// TestByteReverseHintNative tests the ByteReverse hint with known values
func TestByteReverseHintNative(t *testing.T) {
	// Test case 1: 4096
	input := big.NewInt(4096)
	
	// Manually compute what byteReverseHint should return
	// Step 1: Convert big.Int (BE internally) to 32-byte LE
	inputBytes := input.Bytes() // BE: [0x10, 0x00] for 4096
	fmt.Printf("Input: %s\n", input.String())
	fmt.Printf("Input BE bytes: %x\n", inputBytes)
	
	le := make([]byte, 32)
	for i := 0; i < len(inputBytes) && i < 32; i++ {
		le[i] = inputBytes[len(inputBytes)-1-i]
	}
	fmt.Printf("LE 32 bytes: %x\n", le)
	
	// Step 2: Reverse the LE bytes
	reversed := make([]byte, 32)
	for i := 0; i < 32; i++ {
		reversed[i] = le[31-i]
	}
	fmt.Printf("Reversed: %x\n", reversed)
	
	// Step 3: Convert LE back to big.Int (need BE for SetBytes)
	be := make([]byte, 32)
	for i := 0; i < 32; i++ {
		be[i] = reversed[31-i]
	}
	fmt.Printf("BE for SetBytes: %x\n", be)
	
	result := new(big.Int).SetBytes(be)
	fmt.Printf("Result: %s\n", result.String())
	
	// Expected from Rust: We need to compute this
	// 4096 in LE 32 bytes: [00 10 00 00 00 00 ... 00]
	// Reversed: [00 00 ... 00 00 10 00] 
	// As LE int: The byte 0x10 is now at position 30, 0x00 at position 31
	// So value = 0x10 * 256^30 = 16 * 256^30
	
	expected := new(big.Int)
	base := big.NewInt(256)
	exp := big.NewInt(30)
	expected.Exp(base, exp, nil)
	expected.Mul(expected, big.NewInt(16))
	fmt.Printf("Expected (computed): %s\n", expected.String())
	
	if result.Cmp(expected) != 0 {
		t.Errorf("ByteReverse mismatch!\nGot:      %s\nExpected: %s", result.String(), expected.String())
	} else {
		t.Log("ByteReverse matches expected!")
	}
}

// TestTruncate128ReverseNative tests the Truncate128Reverse hint with known values  
func TestTruncate128ReverseNative(t *testing.T) {
	// Test with a large value that has bits in the high 128 bits
	// Use hash of [0,0,0] = 5317387130258456662214331362918410991734007599705406860481038345552731150762
	input := new(big.Int)
	input.SetString("5317387130258456662214331362918410991734007599705406860481038345552731150762", 10)
	
	fmt.Printf("Input: %s\n", input.String())
	
	// Step 1: Convert to 32-byte LE
	inputBytes := input.Bytes() // BE
	fmt.Printf("Input BE bytes (%d): %x\n", len(inputBytes), inputBytes)
	
	le := make([]byte, 32)
	for i := 0; i < len(inputBytes) && i < 32; i++ {
		le[i] = inputBytes[len(inputBytes)-1-i]
	}
	fmt.Printf("LE 32 bytes: %x\n", le)
	
	// Step 2: Take first 16 bytes (128 bits)
	le16 := le[:16]
	fmt.Printf("LE16 (first 16 bytes): %x\n", le16)
	
	// Step 3: Reverse the 16 bytes
	reversed := make([]byte, 16)
	for i := 0; i < 16; i++ {
		reversed[i] = le16[15-i]
	}
	fmt.Printf("Reversed 16 bytes: %x\n", reversed)
	
	// Step 4: Convert LE to big.Int
	be := make([]byte, 16)
	for i := 0; i < 16; i++ {
		be[i] = reversed[15-i]
	}
	fmt.Printf("BE for SetBytes: %x\n", be)
	
	result := new(big.Int).SetBytes(be)
	fmt.Printf("Result: %s\n", result.String())
}
