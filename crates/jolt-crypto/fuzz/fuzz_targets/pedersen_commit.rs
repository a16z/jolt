#![no_main]

//! Pedersen vector-commitment correctness over a fixed generator setup.
//!
//! The setup lives in a `OnceLock`; the fuzzer controls the vector length
//! (exercising the short-input prefix path), every committed value, and both
//! blinding factors. Oracles: commit/verify round-trip, additive
//! homomorphism, and single-position and wrong-blinding must-reject.

use std::sync::OnceLock;

use jolt_crypto::{Bn254, Bn254G1, Pedersen, PedersenSetup, VectorCommitment};
use jolt_field::{Fr, FromPrimitiveInt, ReducingBytes};
use libfuzzer_sys::fuzz_target;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

const MAX_LEN: usize = 8;
const SCALAR_BYTES: usize = 32;

fn setup() -> &'static PedersenSetup<Bn254G1> {
    static SETUP: OnceLock<PedersenSetup<Bn254G1>> = OnceLock::new();
    SETUP.get_or_init(|| {
        let mut rng = ChaCha20Rng::seed_from_u64(0xf022);
        let generators: Vec<Bn254G1> = (0..MAX_LEN).map(|_| Bn254::random_g1(&mut rng)).collect();
        let blinding = Bn254::random_g1(&mut rng);
        PedersenSetup::new(generators, blinding)
    })
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    // Layout: length selector, two value vectors, two blindings, flip index.
    let len = (data[0] as usize % MAX_LEN) + 1; // 1..=8 exercises the prefix path
    if data.len() < 1 + (2 * len + 2) * SCALAR_BYTES + 1 {
        return;
    }
    let scalar_at = |index: usize| {
        let start = 1 + index * SCALAR_BYTES;
        <Fr as ReducingBytes>::from_le_bytes_mod_order(&data[start..start + SCALAR_BYTES])
    };
    let values_a: Vec<Fr> = (0..len).map(scalar_at).collect();
    let values_b: Vec<Fr> = (0..len).map(|i| scalar_at(len + i)).collect();
    let blind_a = scalar_at(2 * len);
    let blind_b = scalar_at(2 * len + 1);
    let flip = data[1 + (2 * len + 2) * SCALAR_BYTES] as usize % len;

    let setup = setup();

    // Commit/verify round-trip.
    let commit_a = Pedersen::<Bn254G1>::commit(setup, &values_a, &blind_a);
    assert!(
        Pedersen::<Bn254G1>::verify(setup, &commit_a, &values_a, &blind_a),
        "commit-verify round-trip failed"
    );

    // Additive homomorphism: C(a, r) + C(b, s) == C(a + b, r + s).
    let commit_b = Pedersen::<Bn254G1>::commit(setup, &values_b, &blind_b);
    let sums: Vec<Fr> = values_a
        .iter()
        .zip(&values_b)
        .map(|(&a, &b)| a + b)
        .collect();
    let commit_sum = Pedersen::<Bn254G1>::commit(setup, &sums, &(blind_a + blind_b));
    assert_eq!(commit_a + commit_b, commit_sum, "homomorphism violated");

    // Binding: one perturbed position or a perturbed blinding must not verify.
    let one = Fr::from_u64(1);
    let mut perturbed = values_a.clone();
    perturbed[flip] += one;
    assert!(
        !Pedersen::<Bn254G1>::verify(setup, &commit_a, &perturbed, &blind_a),
        "verify accepted a perturbed value"
    );
    assert!(
        !Pedersen::<Bn254G1>::verify(setup, &commit_a, &values_a, &(blind_a + one)),
        "verify accepted a perturbed blinding"
    );
});
