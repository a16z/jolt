#![cfg_attr(feature = "guest", no_std)]

use core::hint::black_box;
use jolt::{end_cycle_tracking, start_cycle_tracking};
use ark_ff::{BigInteger, BigInteger256};

/// Multiplies two 256-bit integers (each represented as two u128 limbs) and returns the 512-bit result as 8 u64 limbs
#[jolt::provable(stack_size = 16384, memory_size = 10240, max_trace_length = 65536)]
fn mul_u256(a_lo: u128, a_hi: u128, b_lo: u128, b_hi: u128) -> (u64, u64, u64, u64, u64, u64, u64, u64) {
    // Black box the inputs to prevent optimization
    let a_lo = black_box(a_lo);
    let a_hi = black_box(a_hi);
    let b_lo = black_box(b_lo);
    let b_hi = black_box(b_hi);
    
    // Create BigInteger256 from the input limbs
    // BigInteger256 uses 4 u64 limbs in little-endian order
    // Convert u128 to u64 limbs: each u128 becomes 2 u64s
    let a_limbs = [
        (a_lo & 0xFFFFFFFFFFFFFFFF) as u64,         // a_lo low 64 bits
        ((a_lo >> 64) & 0xFFFFFFFFFFFFFFFF) as u64, // a_lo high 64 bits
        (a_hi & 0xFFFFFFFFFFFFFFFF) as u64,         // a_hi low 64 bits
        ((a_hi >> 64) & 0xFFFFFFFFFFFFFFFF) as u64, // a_hi high 64 bits
    ];
    
    let b_limbs = [
        (b_lo & 0xFFFFFFFFFFFFFFFF) as u64,         // b_lo low 64 bits
        ((b_lo >> 64) & 0xFFFFFFFFFFFFFFFF) as u64, // b_lo high 64 bits
        (b_hi & 0xFFFFFFFFFFFFFFFF) as u64,         // b_hi low 64 bits
        ((b_hi >> 64) & 0xFFFFFFFFFFFFFFFF) as u64, // b_hi high 64 bits
    ];
    
    let a = black_box(BigInteger256::new(a_limbs));
    let b = black_box(BigInteger256::new(b_limbs));
    
    // Multiply using mul - returns (low_bits, high_bits) as two BigInteger256
    start_cycle_tracking("big_int_mul");
    let (low, high) = black_box(a.mul(&b));
    end_cycle_tracking("big_int_mul");
    
    // Get the result limbs (they're stored in little-endian order)
    let low_limbs = low.0;
    let high_limbs = high.0;
    
    // Return all 8 u64 limbs (4 from low, 4 from high)
    (
        black_box(low_limbs[0]), 
        black_box(low_limbs[1]), 
        black_box(low_limbs[2]), 
        black_box(low_limbs[3]),
        black_box(high_limbs[0]),
        black_box(high_limbs[1]),
        black_box(high_limbs[2]),
        black_box(high_limbs[3])
    )
}


