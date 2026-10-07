//! Soundness tests: adversarial proofs against sumcheck verification.
//! A proof is the NARG byte string, so every attack is a byte-level edit of
//! an honest proof or a mismatched public statement.

#![expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "tests may panic on assertion failures and index fixture data"
)]

use jolt_field::{CanonicalBytes, CanonicalDecode, Fr, Ring};
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_sumcheck::{
    send_compressed_round, send_full_round, BooleanHypercube, EvaluationClaim, SumcheckClaim,
    SumcheckError, SumcheckVerifier,
};
use jolt_transcript::{
    Blake2b512, Channel, ProtocolId, ProverTranscript, TranscriptError, VerifierTranscript,
};

type F = Fr;

const PROTOCOL: ProtocolId = ProtocolId::new::<Blake2b512>("jolt-sumcheck/tests/soundness");
const SESSION: &[u8] = b"soundness-test";
/// Bytes per degree-1 full round message.
const ROUND_BYTES: usize = 2 * F::NUM_BYTES;

/// Honest degree-1 prover over HighToLow-bound evaluations, full rounds.
fn honest_prove(evals: &[F]) -> Vec<u8> {
    let mut transcript = ProverTranscript::<Blake2b512>::new(&PROTOCOL, SESSION);
    let mut buf = evals.to_vec();
    while buf.len() > 1 {
        let half = buf.len() / 2;
        let eval_0: F = buf[..half].iter().copied().sum();
        let eval_1: F = buf[half..].iter().copied().sum();
        let round_poly = UnivariatePoly::new(vec![eval_0, eval_1 - eval_0]);
        send_full_round(&round_poly, 1, &mut transcript).unwrap();
        let r: F = transcript.challenge_small();
        for i in 0..half {
            buf[i] = buf[i] + r * (buf[i + half] - buf[i]);
        }
        buf.truncate(half);
    }
    transcript.finish()
}

fn claim(num_vars: usize, claimed_sum: F) -> SumcheckClaim<F> {
    SumcheckClaim::new(num_vars, 1, claimed_sum)
}

fn sum(evals: &[F]) -> F {
    evals.iter().copied().sum()
}

/// Runs the verifier over the whole proof, including the trailing-bytes check.
fn verify(claim: &SumcheckClaim<F>, narg: &[u8]) -> Result<EvaluationClaim<F>, SumcheckError<F>> {
    let mut transcript = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, SESSION, narg);
    let reduced = SumcheckVerifier::verify(claim, BooleanHypercube, &mut transcript)?;
    transcript.finish()?;
    Ok(reduced)
}

fn oracle_accepts(reduced: &EvaluationClaim<F>, intended_evals: &[F]) -> bool {
    reduced.value == Polynomial::new(intended_evals.to_vec()).evaluate_and_consume(&reduced.point)
}

#[test]
fn honest_proofs_pass_the_oracle_check() {
    let tables: [Vec<F>; 3] = [
        (1..=8).map(F::from_u64).collect(),
        vec![F::from_u64(0); 8],
        vec![F::from_u64(7); 8],
    ];
    for evals in &tables {
        let reduced = verify(&claim(3, sum(evals)), &honest_prove(evals)).unwrap();
        assert_eq!(reduced.point.len(), 3);
        assert!(oracle_accepts(&reduced, evals));
    }
    // A constant table reduces to the constant at every point.
    let reduced = verify(&claim(3, sum(&tables[2])), &honest_prove(&tables[2])).unwrap();
    assert_eq!(reduced.value, F::from_u64(7));
}

#[test]
fn wrong_polynomial_same_sum_fails_oracle_check() {
    // An honest proof for g passes every round check against sum(g) = sum(f);
    // only the oracle check against f catches the substitution.
    let f_evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let g_evals: Vec<F> = (1..=8).rev().map(F::from_u64).collect();
    assert_eq!(sum(&f_evals), sum(&g_evals));

    let reduced = verify(&claim(3, sum(&g_evals)), &honest_prove(&g_evals)).unwrap();
    assert!(oracle_accepts(&reduced, &g_evals));
    assert!(!oracle_accepts(&reduced, &f_evals));
}

#[test]
fn wrong_claimed_sum_fails_first_round_check() {
    let evals: Vec<F> = (10..=17).map(F::from_u64).collect();
    let result = verify(
        &claim(3, sum(&evals) + F::from_u64(1)),
        &honest_prove(&evals),
    );
    assert!(matches!(
        result,
        Err(SumcheckError::RoundCheckFailed { round: 0, .. })
    ));
}

#[test]
fn every_byte_flip_is_rejected_at_its_round() {
    // A full degree-1 round is checked through 2*c0 + c1, so any change to
    // either coefficient fails its own round, unless the edit already breaks
    // the canonical encoding.
    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let claim = claim(3, sum(&evals));
    let narg = honest_prove(&evals);
    assert_eq!(narg.len(), 3 * ROUND_BYTES);

    for index in 0..narg.len() {
        for mask in [0x01, 0x80] {
            let mut tampered = narg.clone();
            tampered[index] ^= mask;
            let result = verify(&claim, &tampered);
            let round = index / ROUND_BYTES;
            assert!(
                matches!(
                    result,
                    Err(SumcheckError::RoundCheckFailed { round: r, .. }) if r == round
                ) || matches!(
                    result,
                    Err(SumcheckError::Transcript(TranscriptError::NonCanonical))
                ),
                "flip {mask:#04x} at byte {index}: {result:?}"
            );
        }
    }
}

#[test]
fn non_canonical_coefficient_is_rejected() {
    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let mut narg = honest_prove(&evals);
    narg[ROUND_BYTES..ROUND_BYTES + F::NUM_BYTES].fill(0xff);
    assert!(F::from_bytes_le_checked(&[0xff; F::NUM_BYTES]).is_none());
    assert!(matches!(
        verify(&claim(3, sum(&evals)), &narg),
        Err(SumcheckError::Transcript(TranscriptError::NonCanonical))
    ));
}

#[test]
fn rearranged_round_messages_are_rejected() {
    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let claim = claim(3, sum(&evals));
    let narg = honest_prove(&evals);
    let round = |k: usize| &narg[k * ROUND_BYTES..(k + 1) * ROUND_BYTES];

    let swapped = [round(1), round(0), round(2)].concat();
    assert!(verify(&claim, &swapped).is_err());

    let replayed = [round(0), round(0), round(0)].concat();
    assert!(verify(&claim, &replayed).is_err());
}

#[test]
fn all_zero_rounds_rejected_for_nonzero_sum() {
    let evals: Vec<F> = (1..=4).map(F::from_u64).collect();
    assert!(matches!(
        verify(&claim(2, sum(&evals)), &[0; 2 * ROUND_BYTES]),
        Err(SumcheckError::RoundCheckFailed { round: 0, .. })
    ));
}

#[test]
fn every_truncation_is_rejected() {
    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let claim = claim(3, sum(&evals));
    let narg = honest_prove(&evals);
    for len in 0..narg.len() {
        assert!(
            matches!(
                verify(&claim, &narg[..len]),
                Err(SumcheckError::Transcript(TranscriptError::Truncated))
            ),
            "prefix of {len} bytes"
        );
    }
}

#[test]
fn trailing_bytes_are_rejected() {
    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let claim = claim(3, sum(&evals));
    let narg = honest_prove(&evals);

    let extra_byte = [narg.as_slice(), &[0]].concat();
    let extra_round = [narg.as_slice(), &narg[..ROUND_BYTES]].concat();
    for proof in [extra_byte, extra_round] {
        assert!(matches!(
            verify(&claim, &proof),
            Err(SumcheckError::Transcript(TranscriptError::TrailingBytes))
        ));
    }
}

#[test]
fn verifier_transcript_desync_rejected() {
    // Round 0's check does not depend on any challenge; the desync surfaces
    // when round 1 is checked against the running sum at the wrong challenge.
    let evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let claim = claim(3, sum(&evals));
    let narg = honest_prove(&evals);

    let mut extra_absorb = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, SESSION, &narg);
    extra_absorb.public(&F::from_u64(0xdead));
    let mut other_session = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, b"other", &narg);
    for transcript in [&mut extra_absorb, &mut other_session] {
        assert!(matches!(
            SumcheckVerifier::verify(&claim, BooleanHypercube, transcript),
            Err(SumcheckError::RoundCheckFailed { round: 1, .. })
        ));
    }
}

#[test]
fn num_vars_zero_reads_nothing_and_returns_the_claimed_sum() {
    // No rounds means no soundness from the sumcheck itself: any claimed sum
    // "verifies", and only the oracle check against the constant catches a lie.
    let reduced = verify(&claim(0, F::from_u64(999)), &[]).unwrap();
    assert_eq!(reduced.value, F::from_u64(999));
    assert!(reduced.point.is_empty());

    let evals: Vec<F> = (1..=2).map(F::from_u64).collect();
    assert!(matches!(
        verify(&claim(0, F::from_u64(999)), &honest_prove(&evals)),
        Err(SumcheckError::Transcript(TranscriptError::TrailingBytes))
    ));
}

/// Honest degree-2 compressed prover for `g * h` (both multilinear,
/// HighToLow binding).
fn honest_prove_product_compressed(g_evals: &[F], h_evals: &[F]) -> Vec<u8> {
    let mut transcript = ProverTranscript::<Blake2b512>::new(&PROTOCOL, SESSION);
    let mut g = g_evals.to_vec();
    let mut h = h_evals.to_vec();
    while g.len() > 1 {
        let half = g.len() / 2;
        let mut coefficients = [F::from_u64(0); 3];
        for i in 0..half {
            let (g_lo, g_hi) = (g[i], g[i + half]);
            let (h_lo, h_hi) = (h[i], h[i + half]);
            coefficients[0] += g_lo * h_lo;
            coefficients[1] += g_lo * (h_hi - h_lo) + h_lo * (g_hi - g_lo);
            coefficients[2] += (g_hi - g_lo) * (h_hi - h_lo);
        }
        send_compressed_round(
            &UnivariatePoly::new(coefficients.to_vec()),
            2,
            &mut transcript,
        )
        .unwrap();
        let r: F = transcript.challenge_small();
        for i in 0..half {
            g[i] = g[i] + r * (g[i + half] - g[i]);
            h[i] = h[i] + r * (h[i + half] - h[i]);
        }
        g.truncate(half);
        h.truncate(half);
    }
    transcript.finish()
}

fn verify_compressed(
    claim: &SumcheckClaim<F>,
    narg: &[u8],
) -> Result<EvaluationClaim<F>, SumcheckError<F>> {
    let mut transcript = VerifierTranscript::<Blake2b512>::new(&PROTOCOL, SESSION, narg);
    let reduced = SumcheckVerifier::verify_compressed(claim, &mut transcript)?;
    transcript.finish()?;
    Ok(reduced)
}

#[test]
fn tampered_compressed_coefficients_rejected_only_by_oracle_check() {
    let num_vars = 3;
    let g_evals: Vec<F> = (1..=8).map(F::from_u64).collect();
    let h_evals: Vec<F> = (3..=10).rev().map(F::from_u64).collect();
    let claimed_sum: F = g_evals.iter().zip(&h_evals).map(|(&g, &h)| g * h).sum();
    let claim = SumcheckClaim::new(num_vars, 2, claimed_sum);
    let product_eval = |point: &[F]| {
        Polynomial::new(g_evals.clone()).evaluate_and_consume(point)
            * Polynomial::new(h_evals.clone()).evaluate_and_consume(point)
    };

    let narg = honest_prove_product_compressed(&g_evals, &h_evals);
    assert_eq!(narg.len(), num_vars * 2 * F::NUM_BYTES);
    let honest = verify_compressed(&claim, &narg).unwrap();
    assert_eq!(honest.value, product_eval(&honest.point));

    // Each round sends [c0, c2]; the verifier re-derives c1 from the running
    // sum, so s(0) + s(1) == running_sum holds by construction and the round
    // loop cannot reject. Soundness rests on the final oracle check.
    for coefficient in 0..2 * num_vars {
        let slot = coefficient * F::NUM_BYTES..(coefficient + 1) * F::NUM_BYTES;
        let tampered_value =
            F::from_bytes_le_checked(&narg[slot.clone()]).unwrap() + F::from_u64(1);
        let mut tampered = narg.clone();
        tampered_value.to_bytes_le(&mut tampered[slot]);

        let reduced = verify_compressed(&claim, &tampered).unwrap();
        assert_ne!(
            reduced.value,
            product_eval(&reduced.point),
            "tampered coefficient {coefficient} must fail the oracle check"
        );
    }
}
