#![cfg_attr(feature = "guest", no_std)]

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn blake2(input: &[u8]) -> [u8; 32] {
    // Use Jolt's optimized Blake2b implementation
    let hash = jolt_inlines_blake2::Blake2b::digest(input);
    
    // Split the 64-byte hash into two 32-byte arrays
    let mut first_half = [0u8; 32];
    // let mut second_half = [0u8; 32];
    
    first_half.copy_from_slice(&hash[0..32]);
    // second_half.copy_from_slice(&hash[32..64]);
    
    // (first_half, second_half)
    first_half
}
