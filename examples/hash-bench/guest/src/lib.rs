#![cfg_attr(feature = "guest", no_std)]

use core::hint::black_box;

use jolt::{end_cycle_tracking, start_cycle_tracking};
use sha2::{self as sha2_reference, Digest};
use jolt_inlines_sha2 as sha2_inline;

use sha3 as keccak_reference;
use jolt_inlines_keccak256 as keccak_inline;

use blake2 as blake2_reference;
use jolt_inlines_blake2 as blake2_inline;
use blake3 as blake3_reference;
use jolt_inlines_blake3 as blake3_inline;

const NUM_ITERATATIONS: usize = 10;

/// Deterministic pseudo-random generator that modifies an array in-place
/// Uses a simple Linear Congruential Generator (LCG) algorithm
fn generate_pseudo_random(array: &mut [u8], seed: u32) {
    const A: u32 = 1664525;
    const C: u32 = 1013904223;
    
    let mut state = seed;
    
    for i in 0..array.len() {
        state = state.wrapping_mul(A).wrapping_add(C);
        let value = (state ^ (state >> 16)) as u8;
        array[i] = value;
    }
}

#[jolt::provable(max_output_size= 4096, memory_size= 33554432, stack_size=10485760, max_trace_length = 655360)]
fn hashbench() -> [u8; 32] {

    // let sha2_input = [5u8; 2048];
    // start_cycle_tracking("sha2_reference");
    // let hash = black_box(sha2_reference::Sha256::digest(black_box(&sha2_input)));
    // black_box(hash);
    // end_cycle_tracking("sha2_reference");

    // start_cycle_tracking("sha2_inline");
    // let hash = black_box(sha2_inline::Sha256::digest(black_box(&sha2_input)));
    // black_box(hash);
    // end_cycle_tracking("sha2_inline");

    // let keccak_input = [5u8; 1024];
    // start_cycle_tracking("keccak_reference");
    // let hash_k = black_box(keccak_reference::Keccak256::digest(black_box(&keccak_input)));
    // black_box(hash_k);
    // end_cycle_tracking("keccak_reference");

    // start_cycle_tracking("keccak_inline");
    // let hash_k = black_box(keccak_inline::Keccak256::digest(black_box(&keccak_input)));
    // black_box(hash_k);
    // end_cycle_tracking("keccak_inline");


    // REFERENCE BLAKE2
    let mut blake2_input = [5u8; 128 * NUM_ITERATATIONS];
    generate_pseudo_random(&mut blake2_input, 84);
    start_cycle_tracking("blake2_reference");
    let hash_b = black_box(blake2_reference::Blake2b512::digest(black_box(&blake2_input)));
    black_box(hash_b);
    end_cycle_tracking("blake2_reference");

    // INLINE BLAKE2
    let mut blake2_input = [5u8; 128 * NUM_ITERATATIONS];
    generate_pseudo_random(&mut blake2_input, 84);
    start_cycle_tracking("blake2_inline");
    let hash_b = black_box(blake2_inline::Blake2b::digest(black_box(&blake2_input)));
    black_box(hash_b);
    end_cycle_tracking("blake2_inline");

    // REFERENCE BLAKE3
    let mut blake3_input = [5u8; 64];
    generate_pseudo_random(&mut blake3_input, 42); // Using 42 as seed
    for _ in 0..NUM_ITERATATIONS {
        start_cycle_tracking("blake3_reference");
        let result = black_box(blake3_reference::hash(black_box(&blake3_input)));
        black_box(result);
        end_cycle_tracking("blake3_reference");
        let hash_32 = *result.as_bytes();
        blake3_input[..32].copy_from_slice(&hash_32);
        blake3_input[32..].copy_from_slice(&hash_32);
    }
    black_box(blake3_input);

    // INLINE BLAKE3
    let mut blake3_input = [5u8; 64];
    generate_pseudo_random(&mut blake3_input, 42);
    for _ in 0..NUM_ITERATATIONS {
        start_cycle_tracking("blake3_inline");
        let result: [u8; 32] = black_box(blake3_inline::Blake3::digest(black_box(&blake3_input)));
        black_box(result);
        end_cycle_tracking("blake3_inline");
        blake3_input[..32].copy_from_slice(&result);
        blake3_input[32..].copy_from_slice(&result);
    }
    black_box(blake3_input);

    return [0; 32];
}