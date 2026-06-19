//! Fuzz `SumcheckVerifier::verify` with attacker-controlled claim and round
//! polynomials. The verifier MUST never panic on any input — it must either
//! return `Ok(EvaluationClaim)` or a typed [`SumcheckError`].
//!
//! This is a single-instance verifier panic-guard, mirroring the audit-driven
//! panic-resistance work in `jolt-prover-legacy` (e.g. PR #1408 wrapping the verifier in
//! `catch_unwind` for malformed proofs).

#![no_main]

use jolt_field::{Fr, ReducingBytes};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{BooleanHypercube, SumcheckClaim, SumcheckVerifier};
use jolt_transcript::{Blake2bTranscript, Transcript};
use libfuzzer_sys::fuzz_target;

/// Bytes per BN254 scalar.
const SCALAR_BYTES: usize = 32;

/// Cap on `num_vars` to keep the fuzz iteration cheap. Real sumchecks bind up
/// to ~30 variables, but fuzzing the verifier panic surface only needs a
/// handful of rounds.
const MAX_NUM_VARS: usize = 8;

/// Cap on `degree` to keep round polys small. Real sumchecks use degree
/// 2..=4; we go up to 6 to exercise the high-degree path.
const MAX_DEGREE: usize = 6;

fuzz_target!(|data: &[u8]| {
    // Header: 1 byte num_vars + 1 byte degree + 32 bytes claimed_sum.
    if data.len() < 2 + SCALAR_BYTES {
        return;
    }

    let num_vars = (data[0] as usize) % (MAX_NUM_VARS + 1);
    let degree = ((data[1] as usize) % MAX_DEGREE) + 1; // SumcheckClaim::new requires >= 1
    let claimed_sum = read_scalar(&data[2..2 + SCALAR_BYTES]);
    let claim = SumcheckClaim::new(num_vars, degree, claimed_sum);

    // Each round polynomial has at most `degree + 1` coefficients on the wire.
    // The fuzzer also picks the polynomial *length* (capped at degree+1) so
    // we exercise the verifier's degree-bound + empty-poly handling.
    let mut cursor = 2 + SCALAR_BYTES;
    let mut round_proofs: Vec<UnivariatePoly<Fr>> = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        if cursor >= data.len() {
            return;
        }
        let coeff_count = (data[cursor] as usize) % (degree + 2); // 0..=degree+1
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

    // The verifier must terminate without panicking on any input.
    let mut transcript = Blake2bTranscript::new(b"jolt-sumcheck-fuzz");
    let _ = SumcheckVerifier::verify::<Fr, _, UnivariatePoly<Fr>, _>(
        &claim,
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
