#![cfg_attr(feature = "guest", no_std)]
use core::hint::black_box;
use jolt::{end_cycle_tracking, start_cycle_tracking};

use ark_bn254::{Bn254, Fq12, Fr, G1Projective, G2Projective};
use ark_ec::pairing::Pairing;
use ark_ec::PrimeGroup;
use ark_ff::{Field, One, PrimeField, Zero};

// Helper to create a pseudo-random Fq12 element from a seed
fn fq12_from_seed(seed: u32) -> Fq12 {
    let mut result = Fq12::one();
    for i in 0..12u32 {
        // Use large prime-like values to create more realistic field elements
        let val = Fr::from((seed.wrapping_mul(0x85ebca77) + i.wrapping_mul(0x13503f79)) as u64);
        result *= Fq12::from(val.into_bigint().0[0]);
    }
    result
}

// Helper to create a pseudo-random Fr element from a seed
fn fr_from_seed(seed: u32) -> Fr {
    // Create a large scalar value using bit manipulation
    let large_val = ((seed as u64) << 32) | ((seed as u64).wrapping_mul(0x9e3779b9));
    Fr::from(large_val)
}

// Helper to create a pseudo-random G1 point from a seed
fn g1_from_seed(seed: u32) -> G1Projective {
    let scalar = fr_from_seed(seed);
    G1Projective::generator() * scalar
}

// Helper to create a pseudo-random G2 point from a seed
fn g2_from_seed(seed: u32) -> G2Projective {
    let scalar = fr_from_seed(seed);
    G2Projective::generator() * scalar
}

#[jolt::provable(
    memory_size = 67108864,
    stack_size = 16777216,
    max_trace_length = 100000000
)]
fn ec_bench(iterations: u32) -> u32 {
    let n = iterations as usize;

    // Benchmark Fq12 addition
    start_cycle_tracking("fq12_add");
    let mut sum = Fq12::zero();
    for i in 0..n {
        let a = fq12_from_seed(0x12345678 + i as u32);
        let b = fq12_from_seed(0x87654321 + i as u32);
        sum += black_box(a + b);
    }
    end_cycle_tracking("fq12_add");
    black_box(&sum);

    // Benchmark Fq12 multiplication
    start_cycle_tracking("fq12_mul");
    let mut prod = Fq12::one();
    for i in 0..n {
        let a = fq12_from_seed(0x23456789 + i as u32);
        let b = fq12_from_seed(0x98765432 + i as u32);
        prod *= black_box(a * b);
    }
    end_cycle_tracking("fq12_mul");
    black_box(&prod);

    // Benchmark Fq12 squaring
    start_cycle_tracking("fq12_square");
    let mut sq = Fq12::one();
    for i in 0..n {
        let a = fq12_from_seed(0x34567890 + i as u32);
        sq *= black_box(a.square());
    }
    end_cycle_tracking("fq12_square");
    black_box(&sq);

    // Benchmark Fq12 inversion
    start_cycle_tracking("fq12_inv");
    let mut inv_sum = Fq12::zero();
    for i in 0..n {
        let a = fq12_from_seed(0x45678901 + i as u32); // Large non-zero value
        inv_sum += black_box(a.inverse().unwrap());
    }
    end_cycle_tracking("fq12_inv");
    black_box(&inv_sum);

    // Benchmark Fq12 exponentiation with small exponent
    start_cycle_tracking("fq12_exp_small");
    let mut exp_sum = Fq12::zero();
    for i in 0..n {
        let base = fq12_from_seed(0x56789012 + i as u32);
        let exp = fr_from_seed(0x11111111 + i as u32); // Use larger exponent
        exp_sum += black_box(base.pow(exp.into_bigint()));
    }
    end_cycle_tracking("fq12_exp_small");
    black_box(&exp_sum);

    // Benchmark Fq12 exponentiation with larger exponent
    start_cycle_tracking("fq12_exp_large");
    let mut exp_sum2 = Fq12::zero();
    for i in 0..n {
        let base = fq12_from_seed(0x67890123 + i as u32);
        let exp = fr_from_seed(0xDEADBEEF + i as u32); // Use much larger exponent
        exp_sum2 += black_box(base.pow(exp.into_bigint()));
    }
    end_cycle_tracking("fq12_exp_large");
    black_box(&exp_sum2);

    // Benchmark G1 addition
    start_cycle_tracking("g1_add");
    let mut g1_sum = G1Projective::generator();
    for i in 0..n {
        let a = g1_from_seed(0x11223344 + i as u32);
        let b = g1_from_seed(0x55667788 + i as u32);
        g1_sum += black_box(a + b);
    }
    end_cycle_tracking("g1_add");
    black_box(&g1_sum);

    // Benchmark G1 scalar multiplication
    start_cycle_tracking("g1_scalar_mul");
    let mut g1_prod = G1Projective::generator();
    for i in 0..n {
        let point = g1_from_seed(0x22334455 + i as u32);
        let scalar = fr_from_seed(0x66778899 + i as u32);
        g1_prod += black_box(point * scalar);
    }
    end_cycle_tracking("g1_scalar_mul");
    black_box(&g1_prod);

    // Benchmark G2 addition
    start_cycle_tracking("g2_add");
    let mut g2_sum = G2Projective::generator();
    for i in 0..n {
        let a = g2_from_seed(0x33445566 + i as u32);
        let b = g2_from_seed(0x778899AA + i as u32);
        g2_sum += black_box(a + b);
    }
    end_cycle_tracking("g2_add");
    black_box(&g2_sum);

    // Benchmark G2 scalar multiplication
    start_cycle_tracking("g2_scalar_mul");
    let mut g2_prod = G2Projective::generator();
    for i in 0..n {
        let point = g2_from_seed(0x44556677 + i as u32);
        let scalar = fr_from_seed(0x8899AABB + i as u32);
        g2_prod += black_box(point * scalar);
    }
    end_cycle_tracking("g2_scalar_mul");
    black_box(&g2_prod);

    // Benchmark pairing (Miller loop + final exp)
    start_cycle_tracking("pairing_full");
    let mut pairing_sum = Fq12::zero();
    for i in 0..n {
        let g1 = g1_from_seed(0x78901234 + i as u32);
        let g2 = g2_from_seed(0xABCDEF01 + i as u32);
        let pairing_result = black_box(Bn254::pairing(g1, g2));
        pairing_sum += pairing_result.0;
    }
    end_cycle_tracking("pairing_full");
    black_box(&pairing_sum);

    // Benchmark Miller loop only
    start_cycle_tracking("miller_loop");
    let mut miller_sum = Fq12::zero();
    for i in 0..n {
        let g1 = g1_from_seed(0x89012345 + i as u32);
        let g2 = g2_from_seed(0xBCDEF012 + i as u32);
        let miller_result = black_box(Bn254::miller_loop(g1, g2));
        miller_sum += miller_result.0;
    }
    end_cycle_tracking("miller_loop");
    black_box(&miller_sum);

    // Benchmark final exponentiation
    start_cycle_tracking("final_exp");
    let mut final_exp_sum = Fq12::zero();
    for i in 0..n {
        let miller_out = fq12_from_seed(0x90123456 + i as u32);
        let miller_loop_output = ark_ec::pairing::MillerLoopOutput(miller_out);
        let final_exp_result = black_box(Bn254::final_exponentiation(miller_loop_output));
        if let Some(result) = final_exp_result {
            final_exp_sum += result.0;
        }
    }
    end_cycle_tracking("final_exp");
    black_box(&final_exp_sum);

    // Return something to prevent optimization
    1u32
}
