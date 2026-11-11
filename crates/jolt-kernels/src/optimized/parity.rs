//! Parity-test harness of the optimized tier: a lockstep round runner that
//! drives a reference kernel and an optimized kernel from identical
//! [`ProverInputs`] over identical challenges and asserts byte-equal round
//! polynomials (`UnivariatePoly` wire form) and equal typed output claims.
//!
//! The witness plane is `jolt_witness::testing::with_sample_backend` — a real
//! `TraceBackend` over a canned trace, the only plane constructible without a
//! `jolt-program` dependency. Its known weaknesses are documented on the
//! per-kernel tests.
#![expect(clippy::expect_used, clippy::panic, reason = "test-only module")]

#[cfg(not(feature = "akita"))]
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltPolynomialId};
use jolt_field::{Fr, JoltField, Ring};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
#[cfg(not(feature = "akita"))]
use jolt_witness::JoltWitnessOracle;

use crate::SumcheckKernel;

/// Deterministic "random-looking" challenge stream for parity runs: distinct
/// odd scalars, nothing adversarial (parity is exact for any challenges).
pub(crate) fn synthetic_point(len: usize, seed: u64) -> Vec<Fr> {
    (0..len as u64)
        .map(|index| {
            Fr::from_u64(
                seed.wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(index * 2 + 3),
            )
        })
        .collect()
}

#[derive(Clone, Copy)]
pub(crate) enum ExceptionalEq {
    Zero,
    One,
    ZeroPrefix,
}

impl ExceptionalEq {
    pub(crate) const ALL: [Self; 3] = [Self::Zero, Self::One, Self::ZeroPrefix];

    pub(crate) fn point<F: JoltField>(self, len: usize, first_bind: F) -> Vec<F> {
        let mut point = vec![
            match self {
                Self::One => F::one(),
                _ => F::zero(),
            };
            len
        ];
        if matches!(self, Self::ZeroPrefix) {
            // Solve eq(w, first_bind)=0, then subsequent coordinates still
            // exercise exceptional endpoints under a vanished prefix.
            let w = (first_bind - F::one())
                * (first_bind + first_bind - F::one())
                    .inverse()
                    .expect("fixture binding is not one half");
            *point.last_mut().expect("fixture has cycle rounds") = w;
        }
        point
    }
}

#[cfg(not(feature = "akita"))]
pub(crate) fn probe_one_hot_family(
    witness: &impl JoltWitnessOracle<Fr>,
    family: impl Fn(usize) -> JoltCommittedPolynomial,
    log_t: usize,
) -> (usize, usize) {
    let mut count = 0;
    let mut chunk_bits = 0;
    while let Ok(shape) = witness.shape(JoltPolynomialId::Committed(family(count))) {
        chunk_bits = shape.rows().ilog2() as usize - log_t;
        count += 1;
        assert!(count <= 1 << 10, "runaway one-hot family probe");
    }
    (count, chunk_bits)
}

/// The initial claim of an honest kernel, recovered through its own round
/// check: probe `prove_round` with a zero claim and read the true domain sum
/// off the `RoundCheckFailed` error (an `Ok` means the claim really is zero).
/// `prove_round(None, ..)` binds nothing, so the probe is state-free.
pub(crate) fn probe_input_claim<F: JoltField, R>(
    kernel: &mut dyn SumcheckKernel<F, Relation = R>,
) -> F
where
    R: ConcreteSumcheck<F>,
{
    match kernel.prove_round(None, 0, F::zero()) {
        Ok(_) => F::zero(),
        Err(SumcheckError::RoundCheckFailed { actual, .. }) => actual,
        Err(error) => panic!("input-claim probe failed structurally: {error}"),
    }
}

/// Drive both kernels through every round with shared challenges, asserting
/// byte-equal round polynomials, then finish both kernels for output-claim
/// comparison. `initial_claim` must be the honest input claim (see
/// [`probe_input_claim`]). Fixture-specific nontriviality checks belong in
/// callers: a zero claim can still yield nonzero round polynomials.
pub(crate) fn run_lockstep<F: JoltField, R>(
    reference: &mut dyn SumcheckKernel<F, Relation = R>,
    optimized: &mut dyn SumcheckKernel<F, Relation = R>,
    initial_claim: F,
    challenges: &[F],
) where
    R: ConcreteSumcheck<F>,
{
    let rounds = reference.num_rounds();
    assert_eq!(rounds, optimized.num_rounds(), "round count mismatch");
    assert_eq!(rounds, challenges.len(), "challenge count mismatch");
    assert!(rounds > 0, "zero-round parity run proves nothing");

    let mut claim = initial_claim;
    for round in 0..rounds {
        let bind = round.checked_sub(1).map(|previous| challenges[previous]);
        let reference_poly = reference
            .prove_round(bind, round, claim)
            .unwrap_or_else(|error| panic!("reference round {round}: {error}"));
        let optimized_poly = optimized
            .prove_round(bind, round, claim)
            .unwrap_or_else(|error| panic!("optimized round {round}: {error}"));
        assert_eq!(
            reference_poly.coefficients(),
            optimized_poly.coefficients(),
            "round {round}: wire-form round polynomials diverge"
        );
        claim = reference_poly.evaluate(challenges[round]);
    }
    let last = *challenges.last().expect("at least one round");
    reference.finish_rounds(last).expect("reference finish");
    optimized.finish_rounds(last).expect("optimized finish");
}

pub(crate) fn reject_bad_eq_rounds<F: JoltField>(
    kernel: &mut dyn ProveRounds<F>,
    round: usize,
    claim: F,
) {
    let challenge = F::from_u64(7);
    let total = kernel.num_rounds();
    assert_invalid_eq(kernel.prove_round(Some(challenge), total, claim), total);
    if round + 1 < total {
        let bind = (round > 0).then_some(challenge);
        assert_invalid_eq(kernel.prove_round(bind, round + 1, claim), round + 1);
    }
    let wrong_bind = if round == 0 { Some(challenge) } else { None };
    assert_invalid_eq(kernel.prove_round(wrong_bind, round, claim), round);
    if round > 0 {
        assert_invalid_eq(
            kernel.prove_round(Some(challenge), round - 1, claim),
            round - 1,
        );
    }
    assert_invalid_eq(kernel.finish_rounds(challenge), total);
}

pub(crate) fn reject_repeated_eq_round<F: JoltField>(
    kernel: &mut dyn ProveRounds<F>,
    bind: Option<F>,
    round: usize,
    claim: F,
) {
    assert_invalid_eq(kernel.prove_round(bind, round, claim), round);
}

pub(crate) fn reject_finished_eq_calls<F: JoltField>(
    kernel: &mut dyn ProveRounds<F>,
    challenge: F,
    claim: F,
) {
    let total = kernel.num_rounds();
    assert_invalid_eq(kernel.finish_rounds(challenge), total);
    assert_invalid_eq(kernel.prove_round(None, total, claim), total);
    assert_invalid_eq(kernel.prove_round(Some(challenge), 0, claim), 0);
}

fn assert_invalid_eq<F: JoltField, T>(result: Result<T, SumcheckError<F>>, round: usize) {
    assert!(
        matches!(result, Err(SumcheckError::InvalidEqRoundState { round: got, .. }) if got == round)
    );
}

pub(crate) fn exercise_eq_sequence<F: JoltField>(
    exercised: &mut dyn ProveRounds<F>,
    control: &mut dyn ProveRounds<F>,
    mut claim: F,
    challenges: &[F],
) {
    assert_eq!(exercised.num_rounds(), challenges.len());
    for (round, &challenge) in challenges.iter().enumerate() {
        let bind = round.checked_sub(1).map(|previous| challenges[previous]);
        reject_bad_eq_rounds(exercised, round, claim);
        let mut actual = exercised
            .prove_round(bind, round, claim)
            .expect("next valid call after rejections");
        let mut expected = control
            .prove_round(bind, round, claim)
            .expect("untouched control");
        reject_repeated_eq_round(exercised, bind, round, claim);
        actual.trim_trailing_zeros();
        expected.trim_trailing_zeros();
        assert_eq!(actual, expected);
        claim = expected.evaluate(challenge);
    }
    if let Some(&challenge) = challenges.last() {
        exercised.finish_rounds(challenge).expect("valid finish");
        control.finish_rounds(challenge).expect("control finish");
        reject_finished_eq_calls(exercised, challenge, claim);
    }
}
