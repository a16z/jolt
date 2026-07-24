#![no_main]

//! Differential check of `verify_compressed` against an in-harness reference
//! verifier that decompresses each wire round and evaluates it in full.
//!
//! The compressed wire format omits every round's linear coefficient; the
//! verifier recovers it from the running sum. This harness replays the same
//! wire rounds through (a) `SumcheckVerifier::verify_compressed` and (b) an
//! explicit decompress-then-evaluate loop with identical transcript framing,
//! and requires identical challenges and final claims. A divergence means
//! the c₁-recovery arithmetic disagrees with the polynomial it defines.

use jolt_field::{Fr, ReducingBytes};
use jolt_poly::CompressedPoly;
use jolt_sumcheck::{
    BooleanHypercube, CompressedSumcheckProof, EvaluationClaim, SumcheckClaim, SumcheckVerifier,
    SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, LabelWithCount, Transcript};
use libfuzzer_sys::fuzz_target;

const SCALAR_BYTES: usize = 32;
const MAX_NUM_VARS: usize = 8;
const MAX_DEGREE: usize = 4;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 + SCALAR_BYTES {
        return;
    }
    let num_vars = ((data[0] as usize) % MAX_NUM_VARS) + 1;
    let degree = ((data[1] as usize) % MAX_DEGREE) + 1;
    let claimed_sum = read_scalar(&data[2..2 + SCALAR_BYTES]);
    let claim = SumcheckClaim::new(num_vars, degree, claimed_sum);

    // Wire rounds: `degree` stored coefficients each (c₁ omitted).
    let mut cursor = 2 + SCALAR_BYTES;
    let mut wire_rounds: Vec<CompressedPoly<Fr>> = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        let needed = degree * SCALAR_BYTES;
        if cursor + needed > data.len() {
            return;
        }
        let coeffs: Vec<Fr> = (0..degree)
            .map(|i| read_scalar(&data[cursor + i * SCALAR_BYTES..cursor + (i + 1) * SCALAR_BYTES]))
            .collect();
        cursor += needed;
        wire_rounds.push(CompressedPoly::new(coeffs));
    }

    // Production path.
    let proof = CompressedSumcheckProof {
        round_polynomials: wire_rounds.clone(),
    };
    let mut transcript = Blake2bTranscript::new(b"jolt-sumcheck-diff-fuzz");
    let production = SumcheckVerifier::verify_compressed(
        &claim,
        &proof,
        BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        &mut transcript,
    );

    // Reference: decompress each round with the running-sum hint, evaluate
    // the full polynomial, and mirror the wire transcript framing exactly.
    let mut transcript = Blake2bTranscript::new(b"jolt-sumcheck-diff-fuzz");
    let mut running_sum = claimed_sum;
    let mut challenges: Vec<Fr> = Vec::with_capacity(num_vars);
    for round in &wire_rounds {
        let stored = round.coeffs_except_linear_term();
        transcript.append(&LabelWithCount(
            SUMCHECK_ROUND_TRANSCRIPT_LABEL,
            stored.len() as u64,
        ));
        for coefficient in stored {
            coefficient.append_to_transcript(&mut transcript);
        }
        let r: Fr = transcript.challenge();
        let full = round.decompress(running_sum);
        running_sum = full.evaluate(r);
        challenges.push(r);
    }
    let reference = EvaluationClaim::new(challenges, running_sum);

    let production_claim = production.expect("verify_compressed rejected well-formed wire rounds");
    assert_eq!(
        production_claim, reference,
        "verify_compressed disagrees with decompress-then-evaluate"
    );
});

#[inline]
fn read_scalar(bytes: &[u8]) -> Fr {
    debug_assert_eq!(bytes.len(), SCALAR_BYTES);
    <Fr as ReducingBytes>::from_le_bytes_mod_order(bytes)
}
