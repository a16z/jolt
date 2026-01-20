#![cfg_attr(feature = "guest", no_std)]

use core::ops::Deref;

#[jolt::provable(
    memory_size = 65536,
    max_trace_length = 16777216,
    max_trusted_advice_size = 32768,
    max_untrusted_advice_size = 16384
)]
fn sha2_chain(
    input: [u8; 32],
    num_iters: u32,
    trusted_data: jolt::TrustedAdvice<&[u8]>,
    untrusted_data: jolt::UntrustedAdvice<&[u8]>,
) -> [u8; 32] {
    let mut hash = input;
    for _ in 0..num_iters {
        hash = jolt_inlines_sha2::Sha256::digest(&hash);
    }
    
    // Use advice data as dummy inputs - just XOR a few bytes without hashing
    // This demonstrates the advice mechanism without increasing computational complexity
    let trusted_data = trusted_data.deref();
    if trusted_data.len() >= 32 {
        for i in 0..32 {
            hash[i] ^= trusted_data[i];
        }
    }
    
    let untrusted_data = untrusted_data.deref();
    if untrusted_data.len() >= 32 {
        for i in 0..32 {
            hash[i] ^= untrusted_data[i];
        }
    }
    
    hash
}
