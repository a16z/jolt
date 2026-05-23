//! Fuzz `BatchedSumcheckVerifier::verify` with multiple attacker-controlled
//! claims sharing one stream of round polynomials. The verifier MUST never
//! panic — only return `Ok(EvaluationClaim)` or a typed [`SumcheckError`].
//!
//! Exercises the front-loaded batching path (different `num_vars` per claim,
//! `mul_pow_2` scaling, `EmptyClaims` handling) that single-instance fuzzing
//! does not cover.

#![no_main]

use jolt_field::{Fr, ReducingBytes};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{BatchedSumcheckVerifier, BooleanHypercube, SumcheckClaim};
use jolt_transcript::{Blake2bTranscript, Transcript};
use libfuzzer_sys::fuzz_target;

/// Bytes per BN254 scalar.
const SCALAR_BYTES: usize = 32;

/// Cap on `num_vars` per claim — kept tiny so the matrix of (n_claims,
/// n_rounds, degree) stays in fuzz-budget territory.
const MAX_NUM_VARS: usize = 6;

/// Cap on `degree` per claim.
const MAX_DEGREE: usize = 5;

/// Cap on the number of claims in the batch.
const MAX_NUM_CLAIMS: usize = 4;

fuzz_target!(|data: &[u8]| {
    // Header: 1 byte n_claims + 1 byte global degree.
    if data.len() < 2 {
        return;
    }

    let n_claims = (data[0] as usize) % (MAX_NUM_CLAIMS + 1);
    let degree = ((data[1] as usize) % MAX_DEGREE) + 1;

    let mut cursor = 2;

    // Each claim contributes 1 byte for num_vars + 32 bytes for claimed_sum.
    let mut claims: Vec<SumcheckClaim<Fr>> = Vec::with_capacity(n_claims);
    let mut max_num_vars = 0usize;
    for _ in 0..n_claims {
        if cursor + 1 + SCALAR_BYTES > data.len() {
            return;
        }
        let nv = (data[cursor] as usize) % (MAX_NUM_VARS + 1);
        cursor += 1;
        let claimed_sum = read_scalar(&data[cursor..cursor + SCALAR_BYTES]);
        cursor += SCALAR_BYTES;
        max_num_vars = max_num_vars.max(nv);
        claims.push(SumcheckClaim::new(nv, degree, claimed_sum));
    }

    // Round polynomials: one per round of the merged sumcheck of length
    // `max_num_vars`. Each poly has fuzzer-controlled length 0..=degree+1.
    let mut round_proofs: Vec<UnivariatePoly<Fr>> = Vec::with_capacity(max_num_vars);
    for _ in 0..max_num_vars {
        if cursor >= data.len() {
            return;
        }
        let coeff_count = (data[cursor] as usize) % (degree + 2);
        cursor += 1;
        let needed = coeff_count * SCALAR_BYTES;
        if cursor + needed > data.len() {
            return;
        }
        let coeffs: Vec<Fr> = (0..coeff_count)
            .map(|i| read_scalar(&data[cursor + i * SCALAR_BYTES..cursor + (i + 1) * SCALAR_BYTES]))
            .collect();
        cursor += needed;
        round_proofs.push(UnivariatePoly::new(coeffs));
    }

    // The verifier must terminate without panicking on any input — including
    // `n_claims = 0`, which must surface as `SumcheckError::EmptyClaims`.
    let mut transcript = Blake2bTranscript::new(b"jolt-sumcheck-batched-fuzz");
    let _ = BatchedSumcheckVerifier::verify::<Fr, _, UnivariatePoly<Fr>, _>(
        &claims,
        &round_proofs,
        BooleanHypercube,
        &mut transcript,
    );
});

#[inline]
fn read_scalar(bytes: &[u8]) -> Fr {
    debug_assert_eq!(bytes.len(), SCALAR_BYTES);
    <Fr as ReducingBytes>::from_le_bytes_mod_order(bytes)
}
