#![no_main]

//! Differential check of the hand-written GLV scalar-multiplication paths
//! against plain double-and-add `scalar_mul`.
//!
//! Replaces the old `group_arith` target, which exercised only arkworks
//! pass-throughs: `fixed_base_vector_msm_g1` (2D GLV with a precomputed
//! Shamir table) and `glv_four_scalar_mul` (4D GLV over G2) are Jolt-owned
//! decompositions with zero prior fuzz coverage.

use std::sync::OnceLock;

use jolt_crypto::ec::bn254::glv;
use jolt_crypto::{Bn254, Bn254G1, Bn254G2, JoltGroup};
use jolt_field::{Fr, RandomSampling, ReducingBytes};
use libfuzzer_sys::fuzz_target;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

const SCALAR_BYTES: usize = 32;
const MAX_SCALARS: usize = 8;
const NUM_G2_POINTS: usize = 4;

fn bases() -> &'static (Bn254G1, Vec<Bn254G2>) {
    static BASES: OnceLock<(Bn254G1, Vec<Bn254G2>)> = OnceLock::new();
    BASES.get_or_init(|| {
        let mut rng = ChaCha20Rng::seed_from_u64(0x61_1f);
        let g1 = Bn254::random_g1(&mut rng);
        let g2s: Vec<Bn254G2> = (0..NUM_G2_POINTS)
            .map(|_| Bn254::g2_generator().scalar_mul(&Fr::random(&mut rng)))
            .collect();
        (g1, g2s)
    })
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let count = (data[0] as usize % MAX_SCALARS) + 1; // 1..=8
    if data.len() < 1 + count * SCALAR_BYTES {
        return;
    }
    let scalars: Vec<Fr> = (0..count)
        .map(|i| {
            let start = 1 + i * SCALAR_BYTES;
            <Fr as ReducingBytes>::from_le_bytes_mod_order(&data[start..start + SCALAR_BYTES])
        })
        .collect();

    let (g1_base, g2_points) = bases();

    let glv_g1 = glv::fixed_base_vector_msm_g1(g1_base, &scalars);
    assert_eq!(glv_g1.len(), scalars.len());
    for (index, (result, scalar)) in glv_g1.iter().zip(&scalars).enumerate() {
        assert_eq!(
            *result,
            g1_base.scalar_mul(scalar),
            "2D GLV fixed-base MSM disagrees with scalar_mul at index {index}"
        );
    }

    let glv_g2 = glv::glv_four_scalar_mul(scalars[0], g2_points);
    assert_eq!(glv_g2.len(), g2_points.len());
    for (index, (result, point)) in glv_g2.iter().zip(g2_points).enumerate() {
        assert_eq!(
            *result,
            point.scalar_mul(&scalars[0]),
            "4D GLV scalar mul disagrees with scalar_mul at point {index}"
        );
    }
});
