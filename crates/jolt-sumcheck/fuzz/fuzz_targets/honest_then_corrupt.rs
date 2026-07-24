#![no_main]

//! Honest sumcheck proof over a real MLE product, then one fuzzer-chosen
//! corruption; the verifier must reject.
//!
//! The harness proves `Σ_x A(x)·B(x)` honestly with an in-harness
//! degree-2 prover (LSB-first binding), then corrupts exactly one thing: the
//! claimed sum, one round coefficient, the round count, or the degree bound.
//! Every corruption breaks a check the verifier performs deterministically —
//! a wrong claimed sum or coefficient breaks that round's `s(0) + s(1)`
//! comparison, and shape corruptions break the count/degree checks. As a
//! belt-and-braces discharge, an accept of a false statement only counts as
//! sound if the returned claim matches the true product evaluation.

use jolt_field::{Fr, FromPrimitiveInt, Invertible, ReducingBytes};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{BooleanHypercube, SumcheckClaim, SumcheckVerifier};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Transcript};
use libfuzzer_sys::fuzz_target;
use num_traits::Zero;

const SCALAR_BYTES: usize = 32;
const MAX_NUM_VARS: usize = 5;

/// Evaluates the multilinear extension of `evals` at `point`, with
/// `point[k]` bound to index bit `k` (LSB-first, matching the prover below).
fn mle_eval(evals: &[Fr], point: &[Fr]) -> Fr {
    let one = Fr::from_u64(1);
    let mut sum = Fr::zero();
    for (index, &coeff) in evals.iter().enumerate() {
        let mut weight = one;
        for (k, &p) in point.iter().enumerate() {
            weight *= if (index >> k) & 1 == 1 { p } else { one - p };
        }
        sum += coeff * weight;
    }
    sum
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let num_vars = (data[0] as usize % MAX_NUM_VARS) + 1; // 1..=5
    let n = 1usize << num_vars;
    let corruption = data[1];
    let corruption_round = data[2] as usize % num_vars;
    let corruption_coeff = data[3] as usize % 3;
    // Corruption scalar + the two evaluation tables.
    if data.len() < 4 + (1 + 2 * n) * SCALAR_BYTES {
        return;
    }
    let scalar_at = |index: usize| {
        let start = 4 + index * SCALAR_BYTES;
        <Fr as ReducingBytes>::from_le_bytes_mod_order(&data[start..start + SCALAR_BYTES])
    };
    let corruption_scalar = scalar_at(0);
    let evals_a: Vec<Fr> = (0..n).map(|i| scalar_at(1 + i)).collect();
    let evals_b: Vec<Fr> = (0..n).map(|i| scalar_at(1 + n + i)).collect();

    let true_sum: Fr = evals_a.iter().zip(&evals_b).map(|(&a, &b)| a * b).sum();

    // Honest degree-2 prover, binding the low variable each round and
    // mirroring the verifier's transcript exactly.
    let two_inverse = Fr::from_u64(2).inverse().expect("2 is invertible");
    let mut a = evals_a.clone();
    let mut b = evals_b.clone();
    let mut transcript = Blake2bTranscript::new(b"jolt-sumcheck-corrupt-fuzz");
    let mut rounds: Vec<UnivariatePoly<Fr>> = Vec::with_capacity(num_vars);
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
        let c0 = s0;
        let c2 = (s2 - s1 - s1 + s0) * two_inverse;
        let c1 = s1 - s0 - c2;
        let poly = UnivariatePoly::new(vec![c0, c1, c2]);

        for coefficient in poly.coefficients() {
            coefficient.append_to_transcript(&mut transcript);
        }
        let r: Fr = transcript.challenge();
        for j in 0..half {
            a[j] = a[2 * j] + r * (a[2 * j + 1] - a[2 * j]);
            b[j] = b[2 * j] + r * (b[2 * j + 1] - b[2 * j]);
        }
        a.truncate(half);
        b.truncate(half);
        rounds.push(poly);
    }

    // Apply exactly one corruption.
    let mut claimed_sum = true_sum;
    match corruption % 5 {
        0 => {
            // False statement: honest proof, wrong claimed sum.
            if corruption_scalar.is_zero() {
                return;
            }
            claimed_sum += corruption_scalar;
        }
        1 => {
            // One round coefficient replaced; breaks that round's s(0)+s(1).
            let coefficients = rounds[corruption_round].coefficients();
            if coefficients[corruption_coeff] == corruption_scalar {
                return;
            }
            let mut replaced = coefficients.to_vec();
            replaced[corruption_coeff] = corruption_scalar;
            rounds[corruption_round] = UnivariatePoly::new(replaced);
        }
        2 => {
            rounds.pop();
        }
        3 => {
            let last = rounds.last().cloned().expect("num_vars >= 1");
            rounds.push(last);
        }
        _ => {
            // Degree inflation past the claimed bound.
            let coefficients = rounds[corruption_round].coefficients();
            let mut inflated = coefficients.to_vec();
            inflated.push(corruption_scalar);
            if inflated.len() <= 3 {
                return;
            }
            rounds[corruption_round] = UnivariatePoly::new(inflated);
        }
    }

    let claim = SumcheckClaim::new(num_vars, 2, claimed_sum);
    let mut verifier_transcript = Blake2bTranscript::new(b"jolt-sumcheck-corrupt-fuzz");
    let result = SumcheckVerifier::verify::<Fr, _, UnivariatePoly<Fr>, _>(
        &claim,
        &rounds,
        BooleanHypercube,
        &mut verifier_transcript,
    );

    match result {
        Err(_) => {}
        Ok(final_claim) => {
            // Discharge: an accept is only sound if the reduced claim is
            // actually true of the underlying product.
            let product =
                mle_eval(&evals_a, &final_claim.point) * mle_eval(&evals_b, &final_claim.point);
            assert_eq!(
                final_claim.value, product,
                "verifier accepted a corrupted proof reducing to a false claim"
            );
        }
    }
});
