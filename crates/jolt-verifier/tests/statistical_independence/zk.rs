//! Statistical independence of ZK proofs, over their hiding payloads.
//!
//! Every masked message on the masked tape is m̂ = m + p under a fresh pad,
//! and every pad row commitment carries a fresh blind. Two honest proofs of
//! one statement must therefore differ in every masked value and pad row,
//! the masked values' low bytes must look uniform, and no masked value may
//! repeat within a proof (a repeat means a reused pad).

#![cfg_attr(
    all(feature = "prover-fixtures", feature = "zk"),
    expect(
        clippy::cast_precision_loss,
        clippy::panic,
        reason = "chi-squared statistics are computed in floating point; tests fail loudly"
    )
)]

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use std::collections::BTreeSet;

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_crypto::Bn254G1;
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_field::{CanonicalBytes, Fr};
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use jolt_verifier::JoltProofClaims;

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
use crate::support::verifier_fixtures::{
    fresh_zk_muldiv_case, zk_muldiv_case, ZkVerifierFixtureCase,
};

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
const NUM_BUCKETS: usize = 16;
/// Chi-squared critical value for 15 degrees of freedom at p ~ 1.2e-5.
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
const CHI2_CRITICAL: f64 = 50.0;

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
#[test]
fn zk_proofs_are_independent_over_hiding_messages() {
    let cached = zk_muldiv_case();
    let fresh = fresh_zk_muldiv_case();
    let (cached_values, cached_rows) = hiding_payload(&cached);
    let (fresh_values, fresh_rows) = hiding_payload(&fresh);
    assert_eq!(
        cached_values.len(),
        fresh_values.len(),
        "proofs of one statement share one masked schedule"
    );
    assert_eq!(cached_rows.len(), fresh_rows.len());
    assert!(
        cached_values.len() >= NUM_BUCKETS * 4,
        "too few masked values ({}) for a bucketed test",
        cached_values.len()
    );

    let repeated: Vec<usize> = (0..cached_values.len())
        .filter(|&index| cached_values[index] == fresh_values[index])
        .collect();
    assert!(
        repeated.is_empty(),
        "masked values identical across fresh proofs at {repeated:?}"
    );
    let repeated_rows: Vec<usize> = (0..cached_rows.len())
        .filter(|&index| cached_rows[index] == fresh_rows[index])
        .collect();
    assert!(
        repeated_rows.is_empty(),
        "pad row commitments identical across fresh proofs at {repeated_rows:?}"
    );

    let cached_buckets = low_byte_buckets(&cached_values);
    let fresh_buckets = low_byte_buckets(&fresh_values);
    for (label, buckets) in [("cached", &cached_buckets), ("fresh", &fresh_buckets)] {
        let chi2 = uniform_chi_squared(buckets);
        assert!(
            chi2 < CHI2_CRITICAL,
            "{label} proof: low-byte chi2 {chi2:.2} >= {CHI2_CRITICAL} over {buckets:?}"
        );
    }
    let chi2 = two_sample_chi_squared(&cached_buckets, &fresh_buckets);
    assert!(
        chi2 < CHI2_CRITICAL,
        "cached and fresh low-byte distributions differ: chi2 {chi2:.2}"
    );

    for (label, values) in [("cached", &cached_values), ("fresh", &fresh_values)] {
        let mut seen = BTreeSet::new();
        let repeats: Vec<usize> = (0..values.len())
            .filter(|&index| !seen.insert(values[index].to_bytes_le_vec()))
            .collect();
        assert!(
            repeats.is_empty(),
            "{label} proof repeats a masked value at {repeats:?}"
        );
    }
}

/// A ZK proof's masked tape values and its pad row commitments.
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn hiding_payload(case: &ZkVerifierFixtureCase) -> (Vec<Fr>, Vec<Bn254G1>) {
    let JoltProofClaims::Zk { masked, zk_proof } = &case.proof.claims else {
        panic!("a ZK fixture carries a masked payload");
    };
    (masked.values.clone(), zk_proof.pad_commitment.clone())
}

/// Buckets each value's low canonical byte by its low nibble.
#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn low_byte_buckets(values: &[Fr]) -> [usize; NUM_BUCKETS] {
    let mut buckets = [0; NUM_BUCKETS];
    for value in values {
        let low = value.to_bytes_le_vec().first().copied().unwrap_or_default();
        buckets[usize::from(low) % NUM_BUCKETS] += 1;
    }
    buckets
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn uniform_chi_squared(buckets: &[usize; NUM_BUCKETS]) -> f64 {
    let expected = buckets.iter().sum::<usize>() as f64 / NUM_BUCKETS as f64;
    buckets
        .iter()
        .map(|&observed| (observed as f64 - expected).powi(2) / expected)
        .sum()
}

#[cfg(all(feature = "prover-fixtures", feature = "zk"))]
fn two_sample_chi_squared(a: &[usize; NUM_BUCKETS], b: &[usize; NUM_BUCKETS]) -> f64 {
    let n_a = a.iter().sum::<usize>() as f64;
    let n_b = b.iter().sum::<usize>() as f64;
    let n = n_a + n_b;
    a.iter()
        .zip(b)
        .filter(|(&x, &y)| x + y > 0)
        .map(|(&x, &y)| {
            let pooled = (x + y) as f64;
            let expected_a = pooled * n_a / n;
            let expected_b = pooled * n_b / n;
            (x as f64 - expected_a).powi(2) / expected_a
                + (y as f64 - expected_b).powi(2) / expected_b
        })
        .sum()
}
