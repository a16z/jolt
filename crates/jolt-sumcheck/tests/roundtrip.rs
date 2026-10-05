//! Integration tests: full prover-verifier roundtrips with product compositions.
//!
//! An honest prover writes each round into a `ProverTranscript`, in full or
//! compressed form, and the verifier reads the resulting proof back; the
//! reduced claim must match the product of the factors at the challenge point.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests may panic on assertion failures and index fixture data"
)]

use jolt_field::{Fr, Ring};
use jolt_poly::{EqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{
    send_compressed_round, send_full_round, BooleanHypercube, EvaluationClaim, SumcheckClaim,
    SumcheckVerifier,
};
use jolt_transcript::{Blake2b512, Channel, ProtocolId, ProverTranscript, VerifierTranscript};

type F = Fr;

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-sumcheck/tests/roundtrip");
const SESSION: &[u8] = b"sumcheck-roundtrip";

#[derive(Clone, Copy)]
enum Wire {
    Full,
    Compressed,
}

/// Proves `sum_{x in {0,1}^n} prod_j f_j(x)` for multilinear `f_j` (HighToLow
/// binding) and returns the proof, the claim, and the prover's final sponge
/// fingerprint.
fn prove_product(
    polys: &[Vec<F>],
    num_vars: usize,
    wire: Wire,
) -> (Vec<u8>, SumcheckClaim<F>, [u8; 32]) {
    let degree = polys.len();
    let n = 1 << num_vars;
    assert!(polys.iter().all(|p| p.len() == n));
    let claimed_sum: F = (0..n)
        .map(|i| polys.iter().map(|p| p[i]).product::<F>())
        .sum();

    let mut transcript = ProverTranscript::<Blake2b512>::new(&PROTOCOL, SESSION);
    let mut bufs: Vec<Vec<F>> = polys.to_vec();
    for _round in 0..num_vars {
        let half = bufs[0].len() / 2;
        let points: Vec<(F, F)> = (0..=degree)
            .map(|t| {
                let ft = F::from_u64(t as u64);
                let value = (0..half)
                    .map(|i| {
                        bufs.iter()
                            .map(|buf| buf[i] + ft * (buf[i + half] - buf[i]))
                            .product::<F>()
                    })
                    .sum();
                (ft, value)
            })
            .collect();
        let round_poly = UnivariatePoly::interpolate(&points);
        match wire {
            Wire::Full => send_full_round(&round_poly, degree, &mut transcript),
            Wire::Compressed => send_compressed_round(&round_poly, degree, &mut transcript),
        }
        .unwrap();
        let r: F = transcript.challenge_small();

        for buf in &mut bufs {
            for i in 0..half {
                buf[i] = buf[i] + r * (buf[i + half] - buf[i]);
            }
            buf.truncate(half);
        }
    }

    let fingerprint = transcript.challenge_bytes::<32>();
    (
        transcript.finish(),
        SumcheckClaim::new(num_vars, degree, claimed_sum),
        fingerprint,
    )
}

/// Proves and verifies the product of `polys` in `wire` form, checking the
/// reduced claim against the factors' evaluations at the challenge point.
fn assert_product_roundtrip(polys: &[Vec<F>], num_vars: usize, wire: Wire) {
    let (narg, claim, prover_state) = prove_product(polys, num_vars, wire);

    let mut transcript = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, SESSION, &narg);
    let EvaluationClaim { point, value } = match wire {
        Wire::Full => SumcheckVerifier::verify(&claim, BooleanHypercube, &mut transcript),
        Wire::Compressed => SumcheckVerifier::verify_compressed(&claim, &mut transcript),
    }
    .unwrap();
    assert_eq!(transcript.challenge_bytes::<32>(), prover_state);
    transcript.finish().unwrap();

    assert_eq!(point.len(), num_vars);
    let expected: F = polys
        .iter()
        .map(|p| Polynomial::new(p.clone()).evaluate_and_consume(&point))
        .product();
    assert_eq!(value, expected);
}

fn table(num_vars: usize, map: impl Fn(u64) -> u64) -> Vec<F> {
    (0..1u64 << num_vars).map(|i| F::from_u64(map(i))).collect()
}

#[test]
fn degree1_roundtrip() {
    for num_vars in [1, 3] {
        for wire in [Wire::Full, Wire::Compressed] {
            assert_product_roundtrip(&[table(num_vars, |i| i + 1)], num_vars, wire);
        }
    }
}

#[test]
fn degree2_product_roundtrip() {
    let num_vars = 4;
    let polys = [table(num_vars, |i| i + 1), table(num_vars, |i| i * 3 + 7)];
    for wire in [Wire::Full, Wire::Compressed] {
        assert_product_roundtrip(&polys, num_vars, wire);
    }
}

#[test]
fn degree3_product_roundtrip() {
    let num_vars = 3;
    let polys = [
        table(num_vars, |i| i + 1),
        table(num_vars, |i| i * 5 + 2),
        table(num_vars, |i| i + 7),
    ];
    for wire in [Wire::Full, Wire::Compressed] {
        assert_product_roundtrip(&polys, num_vars, wire);
    }
}

#[test]
fn eq_weighted_sumcheck() {
    // eq(r, x) * f(x), the Spartan outer-sumcheck shape.
    let num_vars = 4;
    let r: Vec<F> = (0..num_vars)
        .map(|i| F::from_u64(i as u64 * 7 + 13))
        .collect();
    let polys = [
        EqPolynomial::evals::<F>(&r, None),
        table(num_vars, |i| i * 3 + 1),
    ];
    for wire in [Wire::Full, Wire::Compressed] {
        assert_product_roundtrip(&polys, num_vars, wire);
    }
}

#[test]
fn large_num_vars_roundtrip() {
    let num_vars = 10;
    let polys = [table(num_vars, |i| i + 1), table(num_vars, |i| i * 7 + 3)];
    assert_product_roundtrip(&polys, num_vars, Wire::Compressed);
}
