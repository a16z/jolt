//! Parity-test harness of the optimized tier: a lockstep round runner that
//! drives a reference kernel and an optimized kernel from identical
//! [`ProverInputs`] over identical challenges and asserts byte-equal round
//! polynomials (`UnivariatePoly` wire form) and equal typed output claims.
//!
//! The in-crate witness-backed tests use
//! `jolt_witness::testing::with_sample_backend`, a real `TraceBackend` over a
//! canned trace. Its known weaknesses are documented on the
//! per-kernel tests.
//!
//! `run_lockstep` compares round coefficients and finishes both kernels; its
//! caller compares output claims. `run_lockstep_checked` also compares canonical
//! opening order and values, validates both derived tables and compares the
//! relation's expected output with the final running claim. Kernel fixtures may
//! supply their own inputs and witness plane.
//!
//! Lock-step comparison of reference and optimized sum-check kernels over any
//! field supported by their relation, including binary fields.
//!
//! The exported helpers are available in tests and with `test-utils`. They
//! panic on failed checks and are intended for kernel tests.
#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "kernel test helpers panic on failed checks by design"
)]

#[cfg(all(test, not(feature = "akita")))]
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltPolynomialId};
use std::fmt::Debug;

use jolt_claims::OutputClaims;
use jolt_field::JoltField;
#[cfg(test)]
use jolt_field::{Fr, Ring};
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::relations::{ConcreteSumcheck, OpeningIdOf, SumcheckOutputClaims};
#[cfg(all(test, not(feature = "akita")))]
use jolt_witness::JoltWitnessOracle;

use crate::{ProverInputs, SumcheckKernel};

/// Deterministic "random-looking" challenge stream for parity runs: distinct
/// odd scalars, nothing adversarial (parity is exact for any challenges).
#[cfg(test)]
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

#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) enum ExceptionalEq {
    Zero,
    One,
    ZeroPrefix,
}

#[cfg(test)]
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

#[cfg(all(test, not(feature = "akita")))]
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

/// The initial claim of an honest reference kernel, recovered through its own round
/// check: probe `prove_round` with a zero claim and read the true domain sum
/// off the `RoundCheckFailed` error (an `Ok` means the claim really is zero).
/// This requires the independent endpoint sum and repeatability conditions below;
/// `prove_round(None, ..)` binds nothing but need not leave kernel state unchanged.
///
/// Probe the first round with a zero claim, returning the actual endpoint sum
/// from `RoundCheckFailed`, or zero when the round succeeds.
///
/// This is the honest input claim only when the first round computes its
/// endpoint sum independently of the supplied claim and the probe can be
/// repeated with the same result. These conditions are not checked; a kernel
/// that recovers an endpoint from the claim may return zero for every probe.
/// Probe the reference kernel used by the lock-step runners.
///
/// # Panics
/// Panics on an error other than `RoundCheckFailed`.
pub fn probe_input_claim<F: JoltField, R>(kernel: &mut dyn SumcheckKernel<F, Relation = R>) -> F
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
///
/// Compare every round's coefficient vector under shared challenges, then
/// finish both kernels. `initial_claim` must be the honest input claim.
///
/// Neither kernel may have bound a challenge; this condition is not checked.
/// Output extraction and relation checks are left to the caller. This helper
/// panics by design and is intended for kernel tests.
///
/// # Panics
/// Panics unless the round counts are equal, positive and equal to
/// `challenges.len()`, or if a round fails, coefficient vectors differ, or
/// finishing either kernel fails. Failures name check (a) and the round.
pub fn run_lockstep<F: JoltField, R>(
    reference: &mut dyn SumcheckKernel<F, Relation = R>,
    optimized: &mut dyn SumcheckKernel<F, Relation = R>,
    initial_claim: F,
    challenges: &[F],
) where
    R: ConcreteSumcheck<F>,
{
    let _ = run_rounds(reference, optimized, initial_claim, challenges);
}

/// Compare rounds and final claims from kernels constructed with `inputs`,
/// returning the reference kernel's output claims for fixture-specific checks.
/// `initial_claim` must be the honest input claim.
///
/// Neither kernel may have bound a challenge, and both must use the supplied
/// inputs; these conditions are not checked. This helper panics by design and
/// is intended for kernel tests.
///
/// # Panics
/// Panics unless (a) round counts are equal, positive and equal to the challenge
/// count, rounds succeed with equal coefficient vectors and both kernels finish;
/// (b) output extraction succeeds on both with equal canonical opening order and
/// opening values; (c) opening-point derivation succeeds and both kernels validate
/// their derived tables; and (d) the relation's expected output equals the final
/// running claim. Every failure names its check and round.
pub fn run_lockstep_checked<F: JoltField, R>(
    inputs: &ProverInputs<'_, F, R>,
    reference: &mut dyn SumcheckKernel<F, Relation = R>,
    optimized: &mut dyn SumcheckKernel<F, Relation = R>,
    initial_claim: F,
    challenges: &[F],
) -> SumcheckOutputClaims<F, R>
where
    R: ConcreteSumcheck<F>,
    SumcheckOutputClaims<F, R>: OutputClaims<F, OpeningIdOf<F, R>>,
    OpeningIdOf<F, R>: PartialEq + Debug,
{
    let claim = run_rounds(reference, optimized, initial_claim, challenges);
    let round = challenges.len() - 1;
    let outputs = reference
        .output_claims(inputs.claims)
        .unwrap_or_else(|error| panic!("check (b), round {round}: reference outputs: {error}"));
    let optimized_outputs = optimized
        .output_claims(inputs.claims)
        .unwrap_or_else(|error| panic!("check (b), round {round}: optimized outputs: {error}"));
    assert_eq!(
        outputs.canonical_order(),
        optimized_outputs.canonical_order(),
        "check (b), round {round}: canonical opening order differs"
    );
    assert_eq!(
        outputs.opening_values(),
        optimized_outputs.opening_values(),
        "check (b), round {round}: opening values differ"
    );
    let output_points = inputs
        .relation
        .derive_opening_points(challenges, inputs.points)
        .unwrap_or_else(|error| panic!("check (c), round {round}: opening points: {error}"));
    reference
        .validate_derived_tables(
            inputs.relation,
            inputs.points,
            &output_points,
            inputs.challenges,
        )
        .unwrap_or_else(|error| {
            panic!("check (c), round {round}: reference derived tables: {error}")
        });
    optimized
        .validate_derived_tables(
            inputs.relation,
            inputs.points,
            &output_points,
            inputs.challenges,
        )
        .unwrap_or_else(|error| {
            panic!("check (c), round {round}: optimized derived tables: {error}")
        });
    let expected = inputs
        .relation
        .expected_output(inputs.points, &outputs, &output_points, inputs.challenges)
        .unwrap_or_else(|error| panic!("check (d), round {round}: expected output: {error}"));
    assert_eq!(
        expected, claim,
        "check (d), round {round}: final claim differs"
    );
    outputs
}

fn run_rounds<F: JoltField, R>(
    reference: &mut dyn SumcheckKernel<F, Relation = R>,
    optimized: &mut dyn SumcheckKernel<F, Relation = R>,
    initial_claim: F,
    challenges: &[F],
) -> F
where
    R: ConcreteSumcheck<F>,
{
    let rounds = reference.num_rounds();
    assert_eq!(
        rounds,
        optimized.num_rounds(),
        "check (a), round 0: round count mismatch"
    );
    assert_eq!(
        rounds,
        challenges.len(),
        "check (a), round 0: challenge count mismatch"
    );
    assert!(rounds > 0, "check (a), round 0: zero-round comparison");

    let mut claim = initial_claim;
    for round in 0..rounds {
        let bind = round.checked_sub(1).map(|previous| challenges[previous]);
        let reference_poly = reference
            .prove_round(bind, round, claim)
            .unwrap_or_else(|error| {
                panic!("check (a), round {round}: reference polynomial: {error}")
            });
        let optimized_poly = optimized
            .prove_round(bind, round, claim)
            .unwrap_or_else(|error| {
                panic!("check (a), round {round}: optimized polynomial: {error}")
            });
        assert_eq!(
            reference_poly.coefficients(),
            optimized_poly.coefficients(),
            "check (a), round {round}: wire-form round polynomials diverge"
        );
        claim = reference_poly.evaluate(challenges[round]);
    }
    let last = *challenges.last().expect("at least one round");
    let round = rounds - 1;
    reference
        .finish_rounds(last)
        .unwrap_or_else(|error| panic!("check (a), round {round}: reference finish: {error}"));
    optimized
        .finish_rounds(last)
        .unwrap_or_else(|error| panic!("check (a), round {round}: optimized finish: {error}"));
    claim
}
