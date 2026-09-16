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
use jolt_sumcheck::SumcheckError;
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

/// Probe the committed one-hot family sizes and chunk bits off the backend's
/// shape surface: family count by scanning indices until the shape errors,
/// chunk bits from `log(one-hot rows) − log_t`.
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
/// byte-equal round polynomials, then finish and return both (fully bound)
/// for output-claim comparison. `initial_claim` must be the honest input
/// claim (see [`probe_input_claim`]); a zero claim is rejected so a
/// degenerate all-zero fixture cannot make the parity vacuous.
pub(crate) fn run_lockstep<F: JoltField, R>(
    reference: &mut dyn SumcheckKernel<F, Relation = R>,
    optimized: &mut dyn SumcheckKernel<F, Relation = R>,
    initial_claim: F,
    challenges: &[F],
) where
    R: ConcreteSumcheck<F>,
{
    assert!(
        initial_claim != F::zero(),
        "zero input claim: the fixture degenerated and parity would be vacuous"
    );
    run_lockstep_degenerate(reference, optimized, initial_claim, challenges);
}

/// [`run_lockstep`] without the nonzero-claim guard, for fixtures whose
/// input claim is HONESTLY zero — the FR kernels' zero-short-circuit paths
/// are exercised by FR-inactive traces where every FR column vanishes, and
/// parity over the (zero) round polynomials is exactly the statement under
/// test. Use `run_lockstep` everywhere else.
pub(crate) fn run_lockstep_degenerate<F: JoltField, R>(
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
