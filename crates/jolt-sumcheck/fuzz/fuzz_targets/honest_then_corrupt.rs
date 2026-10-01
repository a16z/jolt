#![no_main]

//! Honest sumcheck proof over a real MLE product, then one fuzzer-chosen
//! corruption; the verifier must reject.
//!
//! The harness proves `Σ_x A(x)·B(x)` with an in-harness degree-2 prover
//! (LSB-first binding) into a NARG argument string, checks that the honest
//! string verifies, then corrupts exactly one thing: the claimed sum, one
//! round coefficient (rewritten as another canonical scalar), one byte, the
//! string's length (a strict prefix or a duplicated trailing round), or the
//! statement's degree bound or round count. Every corruption breaks a check
//! the verifier performs deterministically: a changed coefficient moves that
//! round's `s(0) + s(1) = 2·c0 + c1 + c2` (each coefficient enters with a
//! nonzero weight), a flipped byte does the same or breaks the canonical
//! encoding, and every length or shape mismatch misframes the fixed-width
//! rounds, ending in `Truncated` or in `finish`'s `TrailingBytes`. So the
//! harness asserts a plain reject: corrupted proofs never verify.

use jolt_field::{CanonicalBytes, CanonicalEncoding, Field, Fr, Ring};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{
    send_full_round, BooleanHypercube, EvaluationClaim, SumcheckClaim, SumcheckError,
    SumcheckVerifier,
};
use jolt_transcript::{Blake2b512, Channel, ProtocolId, ProverTranscript, VerifierTranscript};
use libfuzzer_sys::fuzz_target;
use num_traits::Zero;

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-sumcheck-fuzz/corrupt");
const SESSION: &[u8] = b"jolt-sumcheck-corrupt-fuzz";
const SCALAR_BYTES: usize = 32;
const MAX_NUM_VARS: usize = 5;
const DEGREE: usize = 2;
const ROUND_BYTES: usize = (DEGREE + 1) * SCALAR_BYTES;

fn verify(
    claim: &SumcheckClaim<Fr>,
    narg: &[u8],
) -> Result<EvaluationClaim<Fr>, SumcheckError<Fr>> {
    let mut transcript = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, SESSION, narg);
    let reduced = SumcheckVerifier::verify(claim, BooleanHypercube, &mut transcript)?;
    transcript.finish()?;
    Ok(reduced)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let num_vars = (data[0] as usize % MAX_NUM_VARS) + 1; // 1..=5
    let n = 1usize << num_vars;
    let corruption = data[1];
    let corruption_round = data[2] as usize % num_vars;
    let corruption_coeff = data[3] as usize % (DEGREE + 1);
    // Corruption scalar + the two evaluation tables.
    if data.len() < 4 + (1 + 2 * n) * SCALAR_BYTES {
        return;
    }
    let scalar_at = |index: usize| {
        let start = 4 + index * SCALAR_BYTES;
        <Fr as CanonicalEncoding>::from_bytes_le_reduced(&data[start..start + SCALAR_BYTES])
    };
    let corruption_scalar = scalar_at(0);
    // The corruption scalar's first two raw bytes also pick a byte within
    // the chosen coefficient and an XOR mask.
    let corruption_byte = data[4] as usize % SCALAR_BYTES;
    let corruption_mask = data[5];
    let mut a: Vec<Fr> = (0..n).map(|i| scalar_at(1 + i)).collect();
    let mut b: Vec<Fr> = (0..n).map(|i| scalar_at(1 + n + i)).collect();

    let true_sum: Fr = a.iter().zip(&b).map(|(&a, &b)| a * b).sum();

    // Honest degree-2 prover, binding the low variable each round.
    let two_inverse = Fr::from_u64(2).inverse().expect("2 is invertible");
    let mut prover = ProverTranscript::<Blake2b512>::new(&PROTOCOL, SESSION);
    for _ in 0..num_vars {
        let half = a.len() / 2;
        let mut s0 = Fr::zero();
        let mut s1 = Fr::zero();
        let mut s2 = Fr::zero();
        for j in 0..half {
            let (a0, a1) = (a[2 * j], a[2 * j + 1]);
            let (b0, b1) = (b[2 * j], b[2 * j + 1]);
            s0 += a0 * b0;
            s1 += a1 * b1;
            // s(2) with a(2) = 2·a1 − a0 by multilinearity.
            s2 += (a1 + a1 - a0) * (b1 + b1 - b0);
        }
        let c2 = (s2 - s1 - s1 + s0) * two_inverse;
        let c1 = s1 - s0 - c2;
        let poly = UnivariatePoly::new(vec![s0, c1, c2]);
        send_full_round(&poly, DEGREE, &mut prover).expect("degree-2 round");

        let r: Fr = prover.challenge_small();
        for j in 0..half {
            a[j] = a[2 * j] + r * (a[2 * j + 1] - a[2 * j]);
            b[j] = b[2 * j] + r * (b[2 * j + 1] - b[2 * j]);
        }
        a.truncate(half);
        b.truncate(half);
    }
    let honest = prover.finish();
    assert_eq!(honest.len(), num_vars * ROUND_BYTES);
    let honest_claim = SumcheckClaim::new(num_vars, DEGREE, true_sum);
    verify(&honest_claim, &honest).expect("honest proof must verify");

    let coeff_offset = corruption_round * ROUND_BYTES + corruption_coeff * SCALAR_BYTES;
    let mut claim = honest_claim.clone();
    let mut narg = honest.clone();
    match corruption % 7 {
        0 => {
            // False statement: honest proof, wrong claimed sum.
            if corruption_scalar.is_zero() {
                return;
            }
            claim.claimed_sum += corruption_scalar;
        }
        1 => {
            // One round coefficient replaced by another canonical scalar.
            corruption_scalar.to_bytes_le(&mut narg[coeff_offset..coeff_offset + SCALAR_BYTES]);
            if narg == honest {
                return;
            }
        }
        2 => {
            // Any strict prefix.
            narg.truncate(coeff_offset + corruption_byte);
        }
        3 => {
            // Duplicated last round.
            narg.extend_from_within(narg.len() - ROUND_BYTES..);
        }
        4 => {
            if corruption_mask == 0 {
                return;
            }
            narg[coeff_offset + corruption_byte] ^= corruption_mask;
        }
        5 => {
            // Statement degree bound disagrees with the proof's round width.
            claim.degree = if corruption_mask % 2 == 0 { 1 } else { 3 };
        }
        _ => {
            // Statement round count disagrees with the proof.
            claim.num_vars = if corruption_mask % 2 == 0 {
                num_vars - 1
            } else {
                num_vars + 1
            };
        }
    }

    let result = verify(&claim, &narg);
    assert!(
        result.is_err(),
        "verifier accepted a corrupted proof (class {}): {result:?}",
        corruption % 7,
    );
});
