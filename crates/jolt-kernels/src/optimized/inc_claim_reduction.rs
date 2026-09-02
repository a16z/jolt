//! Optimized increment claim-reduction (stage 6b) kernel, byte-parity twin of
//! [`crate::reference::inc_claim_reduction`].
//!
//! Ported legacy technique
//! (`jolt-prover-legacy/src/zkvm/claim_reductions/increments.rs`):
//! **paired-eq fusion** — the four upstream eq leaves enter the summand
//! linearly per increment column, so they collapse into two combined tables
//!
//! `A(j) = eq(r_ram_rw, j) + γ·eq(r_ram_val, j)`
//! `B(j) = γ²·eq(s_reg_rw, j) + γ³·eq(s_reg_val, j)`
//!
//! and the summand is `A·RamInc + B·RdInc`: four bound tables instead of six
//! and two fused multiplies per point, with field-identical round messages.
//! The legacy prefix/suffix compact-scalar phase machinery (`i128` increment
//! scalars promoted on bind) is NOT portable through the modular witness
//! seam — `oracle_table` serves field elements — so the increment columns
//! stay dense field tables here. Eval-at-1 recovery and rayon walks as per
//! the [`crate::optimized`] module docs.

use jolt_claims::protocols::jolt::geometry::claim_reductions::increments::{
    ram_inc_reduced, rd_inc_reduced,
};
use jolt_field::JoltField;
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{ConcreteSumcheck, SumcheckInputClaims};
use jolt_verifier::stages::stage6b::inc_claim_reduction::{
    IncClaimReduction, IncClaimReductionOutputClaims,
};
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::support::{
    bind_all, pair, par_sum_pair_groups, round_poly_from_skipped_evals, scaled_eq_table,
    RoundProgress,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Stage-6b increment claim reduction: `PrepareKernel` front of the
/// optimized kernel.
pub struct OptimizedIncClaimReduction;

/// The four dense round tables of the paired-eq-fused summand
/// `A·RamInc + B·RdInc`, as built by [`build_inc_tables`] — shared between
/// the optimized kernel and its Metal twin (which serves the same tables
/// from unified-memory buffers).
pub(crate) struct IncTables<F> {
    pub(crate) rounds: usize,
    pub(crate) ram_inc: Vec<F>,
    pub(crate) rd_inc: Vec<F>,
    pub(crate) ram_weights: Vec<F>,
    pub(crate) rd_weights: Vec<F>,
}

pub(crate) fn validate_inc_relation<F: JoltField>(
    relation: &IncClaimReduction<F>,
) -> Result<usize, KernelError<F>> {
    for point in relation.cycle_points() {
        if point.len() != relation.rounds() {
            return Err(KernelError::InvariantViolation {
                reason: "increment reduction cycle point has the wrong variable count",
            });
        }
    }
    Ok(1usize << relation.rounds())
}

pub(crate) fn build_inc_weights<F: JoltField>(
    cycle_points: [&[F]; 4],
    gamma: F,
) -> (Vec<F>, Vec<F>) {
    let gamma_squared = gamma * gamma;
    // A = eq(ram rw) + γ·eq(ram val); B = γ²·eq(reg rw) + γ³·eq(reg val).
    let combine = |first: &[F], first_scale: F, second: &[F], second_scale: F| -> Vec<F> {
        let mut table = scaled_eq_table(first, first_scale);
        let scaled = scaled_eq_table(second, second_scale);
        #[cfg(feature = "parallel")]
        table
            .par_iter_mut()
            .zip(scaled.par_iter())
            .for_each(|(acc, term)| *acc += *term);
        #[cfg(not(feature = "parallel"))]
        table
            .iter_mut()
            .zip(scaled.iter())
            .for_each(|(acc, term)| *acc += *term);
        table
    };
    (
        combine(cycle_points[0], F::one(), cycle_points[1], gamma),
        combine(
            cycle_points[2],
            gamma_squared,
            cycle_points[3],
            gamma_squared * gamma,
        ),
    )
}

pub(crate) fn materialize_inc_columns<F: JoltField>(
    witness: &dyn JoltWitnessPlane<F>,
    cycles: usize,
) -> Result<(Vec<F>, Vec<F>), KernelError<F>> {
    let dense =
        |id: jolt_claims::protocols::jolt::JoltOpeningId| -> Result<Vec<F>, KernelError<F>> {
            let table = witness.oracle_table(id.polynomial_id())?;
            if table.len() != cycles {
                return Err(KernelError::TableSizeMismatch {
                    table: format!("{id:?}"),
                    expected: cycles,
                    got: table.len(),
                });
            }
            Ok(table)
        };
    Ok((dense(ram_inc_reduced())?, dense(rd_inc_reduced())?))
}

pub(crate) fn build_inc_tables<F: JoltField>(
    witness: &dyn JoltWitnessPlane<F>,
    inputs: &ProverInputs<'_, F, IncClaimReduction<F>>,
) -> Result<IncTables<F>, KernelError<F>> {
    let relation = inputs.relation;
    let cycle_points = relation.cycle_points();
    let cycles = validate_inc_relation(relation)?;
    let (ram_weights, rd_weights) = build_inc_weights(cycle_points, inputs.challenges.gamma);
    let (ram_inc, rd_inc) = materialize_inc_columns(witness, cycles)?;

    Ok(IncTables {
        rounds: relation.rounds(),
        ram_inc,
        rd_inc,
        ram_weights,
        rd_weights,
    })
}

impl<F: JoltField> PrepareKernel<F, IncClaimReduction<F>> for OptimizedIncClaimReduction {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, IncClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = IncClaimReduction<F>>>, KernelError<F>> {
        let tables = build_inc_tables(witness, &inputs)?;
        Ok(Box::new(IncKernel {
            progress: RoundProgress::new(tables.rounds),
            ram_inc: Polynomial::new(tables.ram_inc),
            rd_inc: Polynomial::new(tables.rd_inc),
            ram_weights: Polynomial::new(tables.ram_weights),
            rd_weights: Polynomial::new(tables.rd_weights),
        }))
    }
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct IncKernel<F: JoltField> {
    progress: RoundProgress,
    ram_inc: Polynomial<F>,
    rd_inc: Polynomial<F>,
    ram_weights: Polynomial<F>,
    rd_weights: Polynomial<F>,
}

impl<F: JoltField> IncKernel<F> {
    fn bind(&mut self, challenge: F) {
        bind_all(
            [
                &mut self.ram_inc,
                &mut self.rd_inc,
                &mut self.ram_weights,
                &mut self.rd_weights,
            ],
            challenge,
        );
        self.progress.advance();
    }

    /// The summand's evaluations at `t ∈ {0, 2}` summed over group `y`.
    #[inline]
    fn group_evals(&self, y: usize) -> [F; 2] {
        let (ram_lo, ram_hi) = pair(&self.ram_inc, y);
        let (rd_lo, rd_hi) = pair(&self.rd_inc, y);
        let (a_lo, a_hi) = pair(&self.ram_weights, y);
        let (b_lo, b_hi) = pair(&self.rd_weights, y);
        [
            a_lo * ram_lo + b_lo * rd_lo,
            (a_hi + a_hi - a_lo) * (ram_hi + ram_hi - ram_lo)
                + (b_hi + b_hi - b_lo) * (rd_hi + rd_hi - rd_lo),
        ]
    }
}

impl<F: JoltField> ProveRounds<F> for IncKernel<F> {
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
        let half = self.ram_inc.len() / 2;

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

impl<F: JoltField> SumcheckKernel<F> for IncKernel<F> {
    type Relation = IncClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<IncClaimReductionOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        Ok(IncClaimReductionOutputClaims {
            ram_inc: self.ram_inc.evals()[0],
            rd_inc: self.rd_inc.evals()[0],
        })
    }
}

/// Byte parity against the reference kernel over the sample backend: dense
/// committed increment columns with live register and RAM activity.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltPolynomialId};
    use jolt_field::{Fr, Ring};
    use jolt_verifier::stages::stage6b::inc_claim_reduction::{
        IncClaimReductionChallenges, IncClaimReductionInputClaims,
    };
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::JoltWitnessOracle;

    use super::*;
    use crate::optimized::parity::{probe_input_claim, run_lockstep, synthetic_point};
    use crate::ReferenceBackend;

    #[test]
    fn inc_claim_reduction_matches_reference() {
        with_sample_backend(|backend| {
            let log_t = JoltWitnessOracle::<Fr>::shape(
                backend,
                JoltPolynomialId::Committed(JoltCommittedPolynomial::RdInc),
            )
            .unwrap()
            .rows()
            .ilog2() as usize;

            let relation = IncClaimReduction::new(
                TraceDimensions::new(log_t),
                synthetic_point(log_t, 3),
                synthetic_point(log_t, 5),
                synthetic_point(log_t, 7),
                synthetic_point(log_t, 11),
            );
            let challenges = IncClaimReductionChallenges {
                gamma: Fr::from_u64(29),
            };
            let claims = IncClaimReductionInputClaims::<Fr>::default();
            let input_points = IncClaimReductionInputClaims::<Vec<Fr>>::default();

            let mut session = ProofSession::default();
            let mut reference =
                <ReferenceBackend as PrepareKernel<Fr, IncClaimReduction<Fr>>>::prepare(
                    &ReferenceBackend,
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &input_points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            let mut optimized = OptimizedIncClaimReduction
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &input_points,
                        challenges: &challenges,
                    },
                )
                .unwrap();

            let claim = probe_input_claim(reference.as_mut());
            let sumcheck_challenges = synthetic_point(log_t, 401);
            run_lockstep(
                reference.as_mut(),
                optimized.as_mut(),
                claim,
                &sumcheck_challenges,
            );
            assert_eq!(
                reference.output_claims(&claims).unwrap(),
                optimized.output_claims(&claims).unwrap()
            );
        });
    }
}
