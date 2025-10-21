#![cfg_attr(feature = "guest", no_std)]

use core::hint::black_box;

use jolt::{end_cycle_tracking, start_cycle_tracking};

use blake2 as blake2_reference;
use blake3 as blake3_reference;
use jolt_inlines_blake2 as blake2_inline;
use jolt_inlines_blake3 as blake3_inline;
use jolt_inlines_keccak256 as keccak_inline;
use jolt_inlines_sha2 as sha2_inline;
use sha2::{self as sha2_reference, Digest};
use sha3 as keccak_reference;

const INPUT_SIZE: usize = 32_768;
const BLAKE3_INPUT_SIZE: usize = 64;

#[jolt::provable(
    max_output_size = 4096,
    memory_size = 33554432,
    stack_size = 10485760,
    max_trace_length = 20553600
)]
fn hashbench() -> [u8; 32] {
    benchmark_sha2_reference();
    benchmark_sha2_inline();
    benchmark_keccak_reference();
    benchmark_keccak_inline();
    benchmark_blake2_reference();
    benchmark_blake2_inline();
    benchmark_blake3_reference();
    benchmark_blake3_inline();

    return [0; 32];
}

/// Assigns deterministic random-looking values to array
/// Uses a simple Linear Congruential Generator (LCG) algorithm to fill the array
fn assign_random_looking_values(array: &mut [u8], seed: u32) {
    const A: u32 = 1664525;
    const C: u32 = 1013904223;
    let mut state = seed;
    for item in array {
        state = state.wrapping_mul(A).wrapping_add(C);
        let value = (state ^ (state >> 16)) as u8;
        *item = value;
    }
}

fn benchmark_sha2_reference() {
    let mut input = [5u8; INPUT_SIZE];
    assign_random_looking_values(&mut input, 40);
    start_cycle_tracking("sha2_reference");
    let result = black_box(sha2_reference::Sha256::digest(black_box(&input)));
    black_box(result);
    end_cycle_tracking("sha2_reference");
}

fn benchmark_sha2_inline() {
    let mut input = [5u8; INPUT_SIZE];
    assign_random_looking_values(&mut input, 40); // Same seed for fair comparison
    start_cycle_tracking("sha2_inline");
    let result = black_box(sha2_inline::Sha256::digest(black_box(&input)));
    black_box(result);
    end_cycle_tracking("sha2_inline");
}

fn benchmark_keccak_reference() {
    let mut input = [5u8; INPUT_SIZE];
    assign_random_looking_values(&mut input, 30);
    start_cycle_tracking("keccak_reference");
    let result = black_box(keccak_reference::Keccak256::digest(black_box(&input)));
    black_box(result);
    end_cycle_tracking("keccak_reference");
}

fn benchmark_keccak_inline() {
    let mut input = [5u8; INPUT_SIZE];
    assign_random_looking_values(&mut input, 30); // Same seed for fair comparison
    start_cycle_tracking("keccak_inline");
    let result = black_box(keccak_inline::Keccak256::digest(black_box(&input)));
    black_box(result);
    end_cycle_tracking("keccak_inline");
}

fn benchmark_blake2_reference() {
    let mut input = [5u8; INPUT_SIZE];
    assign_random_looking_values(&mut input, 20);
    start_cycle_tracking("blake2_reference");
    let result = black_box(blake2_reference::Blake2b512::digest(black_box(&input)));
    black_box(result);
    end_cycle_tracking("blake2_reference");
}

fn benchmark_blake2_inline() {
    let mut input = [5u8; INPUT_SIZE];
    assign_random_looking_values(&mut input, 20);
    start_cycle_tracking("blake2_inline");
    let result = black_box(blake2_inline::Blake2b::digest(black_box(&input)));
    black_box(result);
    end_cycle_tracking("blake2_inline");
}

fn benchmark_blake3_reference() {
    let mut input = [5u8; BLAKE3_INPUT_SIZE];
    assign_random_looking_values(&mut input, 10);
    start_cycle_tracking("blake3_reference");
    let result = black_box(blake3_reference::hash(black_box(&input)));
    black_box(result);
    end_cycle_tracking("blake3_reference");
}

fn benchmark_blake3_inline() {
    let mut input = [5u8; BLAKE3_INPUT_SIZE];
    assign_random_looking_values(&mut input, 10);
    start_cycle_tracking("blake3_inline");
    let result: [u8; 32] = black_box(blake3_inline::Blake3::digest(black_box(&input)));
    black_box(result);
    end_cycle_tracking("blake3_inline");
}
