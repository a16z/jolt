#!/usr/bin/env python3
"""
Compute the expected output for the big-int example.
This performs 256-bit integer multiplication: a * b
"""

def compute_256bit_multiplication():
    # Input values from the Rust code
    A_LOW = 0xf3a8_9b7c_4d2e_1f0a_8b6c_5d3e_2f1a_0b9c
    A_HIGH = 0x9c8b_7a6f_5e4d_3c2b_1a09_8776_5544_3322
    B_LOW = 0x1234_5678_9abc_def0_fedc_ba98_7654_3210
    B_HIGH = 0xa5b6_c7d8_e9fa_0b1c_2d3e_4f50_6172_8394
    
    # Construct 256-bit integers from the limbs
    # Each 256-bit number = high_limb * 2^128 + low_limb
    a = (A_HIGH << 128) | A_LOW
    b = (B_HIGH << 128) | B_LOW
    
    print("=== Input Values ===")
    print(f"A_LOW:  0x{A_LOW:032x}")
    print(f"A_HIGH: 0x{A_HIGH:032x}")
    print(f"B_LOW:  0x{B_LOW:032x}")
    print(f"B_HIGH: 0x{B_HIGH:032x}")
    print()
    
    print("=== 256-bit Values ===")
    print(f"a = 0x{a:064x}")
    print(f"    = {a}")
    print(f"b = 0x{b:064x}")
    print(f"    = {b}")
    print()
    
    # Perform multiplication
    result = a * b
    
    print(f"Result (a * b): 0x{result:0128x}")
    print(f"                = {result}")
    print()
    
    # The result is a 512-bit number, split into low 256 bits and high 256 bits
    max_256bit = (1 << 256) - 1
    low_256 = result & max_256bit
    high_256 = result >> 256
    
    print("=== Result split into 256-bit parts ===")
    print(f"Low 256 bits:  0x{low_256:064x}")
    print(f"High 256 bits: 0x{high_256:064x}")
    print()
    
    # Convert to u64 digits (little-endian order)
    # Extract eight 64-bit words (4 from low, 4 from high)
    u64_mask = (1 << 64) - 1
    
    # Low 256 bits as 4 u64s
    digit0 = low_256 & u64_mask
    digit1 = (low_256 >> 64) & u64_mask
    digit2 = (low_256 >> 128) & u64_mask
    digit3 = (low_256 >> 192) & u64_mask
    
    # High 256 bits as 4 u64s
    digit4 = high_256 & u64_mask
    digit5 = (high_256 >> 64) & u64_mask
    digit6 = (high_256 >> 128) & u64_mask
    digit7 = (high_256 >> 192) & u64_mask
    
    print("=== Output as u64 digits (little-endian) ===")
    print("Low 256 bits:")
    print(f"  digit[0] (bits 0-63):     0x{digit0:016x} = {digit0}")
    print(f"  digit[1] (bits 64-127):   0x{digit1:016x} = {digit1}")
    print(f"  digit[2] (bits 128-191):  0x{digit2:016x} = {digit2}")
    print(f"  digit[3] (bits 192-255):  0x{digit3:016x} = {digit3}")
    print("High 256 bits:")
    print(f"  digit[4] (bits 256-319):  0x{digit4:016x} = {digit4}")
    print(f"  digit[5] (bits 320-383):  0x{digit5:016x} = {digit5}")
    print(f"  digit[6] (bits 384-447):  0x{digit6:016x} = {digit6}")
    print(f"  digit[7] (bits 448-511):  0x{digit7:016x} = {digit7}")
    print()
    
    # Verify by reconstructing
    reconstructed_low = digit0 | (digit1 << 64) | (digit2 << 128) | (digit3 << 192)
    reconstructed_high = digit4 | (digit5 << 64) | (digit6 << 128) | (digit7 << 192)
    reconstructed_full = reconstructed_low | (reconstructed_high << 256)
    
    print("=== Verification ===")
    print(f"Reconstructed low:  0x{reconstructed_low:064x}")
    print(f"Reconstructed high: 0x{reconstructed_high:064x}")
    print(f"Reconstructed full: 0x{reconstructed_full:0128x}")
    print(f"Matches result: {reconstructed_full == result}")
    
    return digit0, digit1, digit2, digit3, digit4, digit5, digit6, digit7

if __name__ == "__main__":
    digits = compute_256bit_multiplication()
    print()
    print("=== Summary ===")
    print(f"Expected output tuple: ({digits[0]}, {digits[1]}, {digits[2]}, {digits[3]}, {digits[4]}, {digits[5]}, {digits[6]}, {digits[7]})")
