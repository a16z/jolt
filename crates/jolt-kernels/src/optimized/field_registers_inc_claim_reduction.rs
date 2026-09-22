//! The optimized field-registers increment claim-reduction (stage 6b)
//! kernel: [`super::inc_claim_reduction`]'s paired-eq fusion at the FR
//! geometry, byte-parity twin of
//! [`crate::reference::field_registers_inc_claim_reduction`].
//!
//! The two upstream eq leaves enter the summand linearly per increment, so
//! they collapse into one combined table
//! `W(j) = eq(r_field_rw, j) + γ·eq(r_field_val, j)` and the summand is
//! `W · FieldRdInc`: two bound tables instead of the reference's three, one
//! fused multiply per point, field-identical round messages. Eval-at-1
//! recovery and rayon walks as per the [`crate::optimized`] module docs.

use jolt_claims::protocols::field_inline::{
    FieldInlineChallengeId, FieldInlineCommittedPolynomial, FieldInlineDerivedId,
    FieldInlinePolynomialId, FieldRegistersIncClaimReductionChallenge,
    FieldRegistersIncClaimReductionPublic,
};
use jolt_claims::SumcheckChallenges as _;
use jolt_field::JoltField;
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage6b::field_registers_inc_claim_reduction::FieldRegistersIncClaimReduction;
use jolt_verifier::VerifierError;
use jolt_witness::{JoltWitnessPlane, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::support::{
    bind_all, pair, par_sum_pair_groups, round_poly_from_skipped_evals, scaled_eq_table,
    RoundProgress,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

pub struct OptimizedFieldRegistersIncClaimReduction;

impl<F: JoltField> PrepareKernel<F, FieldRegistersIncClaimReduction<F>>
    for OptimizedFieldRegistersIncClaimReduction
{
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, FieldRegistersIncClaimReduction<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = FieldRegistersIncClaimReduction<F>>>,
        KernelError<F>,
    > {
        let relation = inputs.relation;
        let [read_write_cycle, val_evaluation_cycle] = relation.cycle_points();
        for point in [read_write_cycle, val_evaluation_cycle] {
            if point.len() != relation.rounds() {
                return Err(KernelError::InvariantViolation {
                    reason: "FR increment reduction cycle point has the wrong variable count",
                });
            }
        }
        let cycles = 1usize << relation.rounds();

        let field_inline =
            witness
                .field_inline()
                .ok_or(KernelError::Witness(WitnessError::UnavailableView {
                    label: "field-registers increment claim-reduction field-inline oracle",
                }))?;
        let inc_table = field_inline.oracle_table(FieldInlinePolynomialId::Committed(
            FieldInlineCommittedPolynomial::FieldRdInc,
        ))?;
        if inc_table.len() != cycles {
            return Err(KernelError::TableSizeMismatch {
                table: "FieldRdInc".to_owned(),
                expected: cycles,
                got: inc_table.len(),
            });
        }
        let gamma = inputs
            .challenges
            .resolve_challenge(&FieldInlineChallengeId::from(
                FieldRegistersIncClaimReductionChallenge::Gamma,
            ))
            .ok_or(KernelError::InvariantViolation {
                reason: "FR increment claim reduction is missing its gamma challenge",
            })?;

        // W = eq(r_field_rw) + γ·eq(r_field_val).
        let mut weights = scaled_eq_table(read_write_cycle, F::one());
        let scaled = scaled_eq_table(val_evaluation_cycle, gamma);
        #[cfg(feature = "parallel")]
        weights
            .par_iter_mut()
            .zip(scaled.par_iter())
            .for_each(|(acc, term)| *acc += *term);
        #[cfg(not(feature = "parallel"))]
        weights
            .iter_mut()
            .zip(scaled.iter())
            .for_each(|(acc, term)| *acc += *term);

        Ok(Box::new(FieldIncKernel {
            progress: RoundProgress::new(relation.rounds()),
            inc: Polynomial::new(inc_table),
            weights: Polynomial::new(weights),
        }))
    }
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct FieldIncKernel<F: JoltField> {
    progress: RoundProgress,
    inc: Polynomial<F>,
    weights: Polynomial<F>,
}

impl<F: JoltField> FieldIncKernel<F> {
    fn bind(&mut self, challenge: F) {
        bind_all([&mut self.inc, &mut self.weights], challenge);
        self.progress.advance();
    }

    /// The summand's evaluations at `t ∈ {0, 2}` summed over group `y`.
    #[inline]
    fn group_evals(&self, y: usize) -> [F; 2] {
        let (inc_lo, inc_hi) = pair(&self.inc, y);
        let (w_lo, w_hi) = pair(&self.weights, y);
        [
            w_lo * inc_lo,
            (w_hi + w_hi - w_lo) * (inc_hi + inc_hi - inc_lo),
        ]
    }
}

impl<F: JoltField> ProveRounds<F> for FieldIncKernel<F> {
    fn num_rounds(&self) -> usize {
        self.progress.total()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let half = self.inc.len() / 2;

        let evals = par_sum_pair_groups(half, 2, |acc, y| {
            let group = self.group_evals(y);
            acc[0] += group[0];
            acc[1] += group[1];
        });

        Ok(round_poly_from_skipped_evals(&evals, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for FieldIncKernel<F> {
    type Relation = FieldRegistersIncClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>> {
        use jolt_claims::protocols::field_inline::relations::claim_reductions::increments::FieldRegistersIncClaimReductionOutputClaims;

        self.progress.require_complete()?;
        Ok(FieldRegistersIncClaimReductionOutputClaims {
            rd_inc: self.inc.evals()[0],
        })
    }

    /// The eq-table cross-checks: the fused weight table is the two eq
    /// leaves' γ-combination, so its bound value must equal
    /// `EqReadWrite + γ·EqValEvaluation` from the verifier's own
    /// `derive_output_term` — except γ is not observable here, so both leaves
    /// pin individually against the SplitEq-free reconstruction: the fused
    /// scalar equals the γ-weighted sum of the two derived terms.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let derive = |public: FieldRegistersIncClaimReductionPublic| {
            relation.derive_output_term(
                &FieldInlineDerivedId::from(public),
                input_points,
                output_points,
                challenges,
            )
        };
        let expected = derive(FieldRegistersIncClaimReductionPublic::EqReadWrite)?
            + challenges.gamma * derive(FieldRegistersIncClaimReductionPublic::EqValEvaluation)?;
        let got = self.weights.evals()[0];
        if got != expected {
            return Err(SumcheckKernelError::Verifier(
                VerifierError::StageClaimSumcheckFailed {
                    stage: "FieldRegistersIncClaimReduction".to_string(),
                    reason: format!(
                        "fused eq table bound to {got:?}, but the derived terms fold to \
                         {expected:?}"
                    ),
                },
            ));
        }
        Ok(())
    }
}

/// Byte parity against the reference kernel on register-consistent FR
/// traces, plus the FR-inactive degenerate case (an all-zero increment
/// column).
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::field_inline::relations::claim_reductions::increments::{
        FieldRegistersIncClaimReductionChallenges, FieldRegistersIncClaimReductionInputClaims,
    };
    use jolt_claims::protocols::field_inline::FieldRegistersTraceDimensions;
    use jolt_field::{Fr, Ring};

    use super::*;
    use crate::optimized::field_registers_testing::{
        inactive_fr_fixture, structured_fr_fixture, FrTraceFixture,
    };
    use crate::optimized::parity::{
        probe_input_claim, run_lockstep, run_lockstep_degenerate, synthetic_point,
    };
    use crate::ReferenceBackend;

    fn run_parity(fixture: FrTraceFixture, log_t: usize, seed: u64, expect_active: bool) {
        fixture.with_plane(log_t, |backend| {
            let relation = FieldRegistersIncClaimReduction::<Fr>::new(
                FieldRegistersTraceDimensions::new(log_t),
                synthetic_point(log_t, seed ^ 0xA5A5),
                synthetic_point(log_t, seed ^ 0x5A5A),
            );
            let claims = FieldRegistersIncClaimReductionInputClaims {
                rd_inc_read_write: Fr::from_u64(0),
                rd_inc_val_evaluation: Fr::from_u64(0),
            };
            let points = FieldRegistersIncClaimReductionInputClaims {
                rd_inc_read_write: Vec::new(),
                rd_inc_val_evaluation: Vec::new(),
            };
            let challenges = FieldRegistersIncClaimReductionChallenges {
                gamma: Fr::from_u64(37 + seed),
            };
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };

            let mut session = ProofSession::default();
            let mut reference = <ReferenceBackend as PrepareKernel<
                Fr,
                FieldRegistersIncClaimReduction<Fr>,
            >>::prepare(
                &ReferenceBackend, &mut session, backend, inputs()
            )
            .unwrap();
            let mut optimized = OptimizedFieldRegistersIncClaimReduction
                .prepare(&mut session, backend, inputs())
                .unwrap();

            let claim = probe_input_claim(reference.as_mut());
            let round_challenges =
                synthetic_point(relation.rounds(), seed.wrapping_mul(0x9E37_79B9));
            if expect_active {
                assert!(claim != Fr::from_u64(0), "FR-active fixture degenerated");
                run_lockstep(
                    reference.as_mut(),
                    optimized.as_mut(),
                    claim,
                    &round_challenges,
                );
            } else {
                assert_eq!(claim, Fr::from_u64(0), "FR-inactive claim must be zero");
                run_lockstep_degenerate(
                    reference.as_mut(),
                    optimized.as_mut(),
                    claim,
                    &round_challenges,
                );
            }
            assert_eq!(
                reference.output_claims(&claims).unwrap(),
                optimized.output_claims(&claims).unwrap()
            );
            let output_points = relation
                .derive_opening_points(&round_challenges, &points)
                .unwrap();
            reference
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
            optimized
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
        });
    }

    #[test]
    fn parity_structured_even_log_t() {
        run_parity(structured_fr_fixture(16), 4, 301, true);
    }

    #[test]
    fn parity_structured_odd_log_t() {
        run_parity(structured_fr_fixture(8), 3, 307, true);
    }

    #[test]
    fn parity_inactive_trace_is_degenerate() {
        run_parity(inactive_fr_fixture(4), 3, 311, false);
    }
}
