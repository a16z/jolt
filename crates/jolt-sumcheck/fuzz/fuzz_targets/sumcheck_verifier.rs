//! Fuzz `SumcheckVerifier::verify_compressed` — the production wire path —
//! with an attacker-controlled claim and compressed round polynomials. The
//! verifier MUST never panic on any input: it must either return
//! `Ok(EvaluationClaim)` or a typed [`SumcheckError`].
//!
//! This exercises the c₁-recovery arithmetic (`evaluate_with_hint`) and the
//! `CompressedPolynomialTooShort` and degree-bound rejects that the clear
//! path never reaches. Full-depth accept-path coverage lives in
//! `valid_prefix_proof`.

#![no_main]

use jolt_field::{Fr, ReducingBytes};
use jolt_poly::CompressedPoly;
use jolt_sumcheck::{
    BooleanHypercube, CompressedSumcheckProof, SumcheckClaim, SumcheckVerifier,
    SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
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

    // A compressed round stores `degree` coefficients (the linear term is
    // omitted and recovered from the running sum). The fuzzer picks each
    // round's stored length (0..=degree+1) so we exercise the too-short and
    // degree-bound rejects around the recovery path.
    let mut cursor = 2 + SCALAR_BYTES;
    let mut round_polynomials: Vec<CompressedPoly<Fr>> = Vec::with_capacity(num_vars);
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
        round_polynomials.push(CompressedPoly::new(coeffs));
    }
    let proof = CompressedSumcheckProof { round_polynomials };

    // The verifier must terminate without panicking on any input.
    let mut transcript = Blake2bTranscript::new(b"jolt-sumcheck-fuzz");
    let _ = SumcheckVerifier::verify_compressed(
        &claim,
        &proof,
        BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        &mut transcript,
    );
});

#[inline]
fn read_scalar(bytes: &[u8]) -> Fr {
    debug_assert_eq!(bytes.len(), SCALAR_BYTES);
    <Fr as ReducingBytes>::from_le_bytes_mod_order(bytes)
}
