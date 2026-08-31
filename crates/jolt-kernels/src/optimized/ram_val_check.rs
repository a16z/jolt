//! The optimized RAM value-check (stage 4) kernel.
//!
//! The summand is `inc(j) · ra(j) · (LT(j, r_cycle) + γ)` over the cycle
//! domain. Techniques ported from `jolt-prover-legacy/src/zkvm/ram/
//! val_evaluation.rs` (the same shape as the optimized registers
//! value-evaluation kernel):
//!
//! - **Lazy one-hot `ra`** ([`LazyFoldedRa`]): rounds 0–3 serve
//!   `ra(j) = eq(r_address)[addresses[j]]` straight from the session-shared
//!   RAM access columns and the `K`-sized eq table — the `(K × T)` grid and
//!   the reference's dense `T`-sized address fold are never materialized;
//!   a dense vector first appears at `T/16`.
//! - **Split LT with γ folded in** ([`SplitLt`]): `LT(j, r_cycle) + γ` is
//!   served from three `~√T` tables instead of a dense `T`-sized one.
//! - **Eval-at-{0,2,3} sampling** with the engine hint supplying `s(1)`.
//! - **Deferred-reduction accumulation** of the triple products.

use jolt_claims::protocols::jolt::geometry::ram::ram_inc_val_check;
use jolt_claims::protocols::jolt::{JoltDerivedId, RamValCheckPublic};
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage4::ram_val_check::{RamValCheck, RamValCheckOutputClaims};
use jolt_witness::JoltWitnessPlane;
use std::sync::Arc;

use super::lazy_ra::{ChunkIndexSource, LazyFoldedRa};
use super::ram_trace::{RamAccessColumns, NO_ACCESS};
use super::support::{pin_derived_term, triple_product_round_evals, RoundProgress, SplitLt};
use super::OptimizedBackend;
use crate::reference::views::eq_table;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Lazy-RA index source over the full remapped RAM address (a single
/// `K`-ary selector), cold on no-access cycles.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct RamAddressIndices {
    columns: Arc<RamAccessColumns>,
}

impl ChunkIndexSource for RamAddressIndices {
    fn num_polys(&self) -> usize {
        1
    }

    fn cycles(&self) -> usize {
        self.columns.addresses.len()
    }

    #[inline]
    fn index(&self, _i: usize, j: usize) -> Option<usize> {
        let address = self.columns.addresses[j];
        (address != NO_ACCESS).then_some(address as usize)
    }
}

impl<F: JoltField> PrepareKernel<F, RamValCheck<F>> for OptimizedBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamValCheck<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamValCheck<F>>>, KernelError<F>> {
        Ok(Box::new(prepare_optimized_ram_val_check(
            session, witness, inputs,
        )?))
    }
}

pub(crate) fn prepare_optimized_ram_val_check<F: JoltField>(
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    inputs: ProverInputs<'_, F, RamValCheck<F>>,
) -> Result<RamValCheckKernel<F>, KernelError<F>> {
    let relation = inputs.relation;
    let log_t = relation.trace_dimensions().log_t();
    let ram_log_k = relation.ram_log_k();
    let ram_val_point: &[F] = &inputs.points.ram_val;
    if ram_val_point.len() != ram_log_k + log_t {
        return Err(KernelError::InvariantViolation {
            reason: "RAM value-check input point has the wrong variable count",
        });
    }
    let (r_address, r_cycle) = ram_val_point.split_at(ram_log_k);

    let columns = RamAccessColumns::shared(session, witness, log_t)?;
    columns.validate_addresses(1usize << ram_log_k)?;

    let inc_table: Vec<F> = witness.oracle_table(ram_inc_val_check().polynomial_id())?;
    if inc_table.len() != 1usize << log_t {
        return Err(KernelError::TableSizeMismatch {
            table: format!("{:?}", ram_inc_val_check()),
            expected: 1usize << log_t,
            got: inc_table.len(),
        });
    }

    Ok(RamValCheckKernel {
        progress: RoundProgress::new(log_t),
        inc: Polynomial::new(inc_table),
        ra: LazyFoldedRa::new(vec![eq_table(r_address)], RamAddressIndices { columns }),
        lt: SplitLt::new_plus_constant(r_cycle, inputs.challenges.gamma),
    })
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
pub(crate) struct RamValCheckKernel<F: JoltField> {
    progress: RoundProgress,
    inc: Polynomial<F>,
    ra: LazyFoldedRa<F, RamAddressIndices>,
    lt: SplitLt<F>,
}

impl<F: JoltField> RamValCheckKernel<F> {
    fn bind(&mut self, challenge: F) {
        self.inc.bind_with_order(challenge, BindingOrder::LowToHigh);
        self.ra.bind(challenge);
        self.lt.bind(challenge);
        self.progress.advance();
    }
}

impl<F: JoltField> ProveRounds<F> for RamValCheckKernel<F> {
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
        let inc = self.inc.evals();
        let evals = triple_product_round_evals(
            half,
            |y| (inc[2 * y], inc[2 * y + 1]),
            |y| self.ra.lo_hi(0, y),
            |y| self.lt.pair(y),
        );

        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for RamValCheckKernel<F> {
    type Relation = RamValCheck<F>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<RamValCheckOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        // The advice cells are dual-role: never bound here, their wire output
        // value is the consumed input claim read back (the naive tier's echo).
        Ok(RamValCheckOutputClaims {
            untrusted_advice: inputs.untrusted_advice,
            trusted_advice: inputs.trusted_advice,
            program_image: inputs.program_image,
            ram_ra: self.ra.final_values()[0],
            ram_inc: self.inc.evals()[0],
        })
    }

    /// Pin the split-LT tables to the verifier's scalar path: the fully bound
    /// `LT + γ` value must equal `derive_output_term(LtCyclePlusGamma)`.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let id = JoltDerivedId::from(RamValCheckPublic::LtCyclePlusGamma);
        pin_derived_term(
            relation,
            id,
            input_points,
            output_points,
            challenges,
            self.lt.final_value(),
        )
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module: fail loudly")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::geometry::ram::{ram_ra_val_check, RamValCheckInit};
    use jolt_claims::protocols::jolt::relations::ram::{
        RamValCheckChallenges, RamValCheckInputClaims,
    };
    use jolt_field::{Fr, Ring};
    use jolt_poly::LtPolynomial;

    use super::super::testing::{
        assert_parity, random_scalars, with_ram_fixture, FixtureShape, RamOp,
    };
    use super::*;
    use crate::reference::views::address_fold;
    use crate::ReferenceBackend;

    fn run_parity(shape: FixtureShape, ops: Vec<RamOp>, seed: u64) {
        with_ram_fixture(shape, ops, |witness| {
            let r_address = random_scalars(shape.log_k(), seed);
            let r_cycle = random_scalars(shape.log_t, seed ^ 0x11);
            let gamma = random_scalars(1, seed ^ 0x22)[0];
            let relation = RamValCheck::<Fr>::new(
                TraceDimensions::new(shape.log_t),
                shape.log_k(),
                RamValCheckInit::full(Fr::from_u64(0)),
            );
            let claims = RamValCheckInputClaims {
                ram_val: Fr::from_u64(0),
                ram_val_final: Fr::from_u64(0),
                untrusted_advice: None,
                trusted_advice: None,
                program_image: None,
            };
            let points = RamValCheckInputClaims::<Vec<Fr>> {
                ram_val: [r_address.clone(), r_cycle.clone()].concat(),
                ram_val_final: r_address.clone(),
                untrusted_advice: None,
                trusted_advice: None,
                program_image: None,
            };
            let challenges = RamValCheckChallenges { gamma };

            let mut reference_session = ProofSession::default();
            let reference = PrepareKernel::<Fr, _>::prepare(
                &ReferenceBackend,
                &mut reference_session,
                witness,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            )
            .unwrap();
            let mut session = ProofSession::default();
            let optimized = PrepareKernel::<Fr, _>::prepare(
                &OptimizedBackend,
                &mut session,
                witness,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            )
            .unwrap();

            // The independently folded true input claim:
            // `Σ_j inc(j) · ra_folded(j) · (LT(j, r_cycle) + γ)`.
            let ra_folded =
                address_fold::<Fr>(witness, ram_ra_val_check(), shape.log_t, &r_address).unwrap();
            let inc: Vec<Fr> = witness
                .oracle_table(ram_inc_val_check().polynomial_id())
                .unwrap();
            let lt = LtPolynomial::evaluations(&r_cycle);
            let input_claim = (0..1usize << shape.log_t)
                .map(|j| inc[j] * ra_folded[j] * (lt[j] + gamma))
                .sum();

            assert_parity(
                reference,
                optimized,
                input_claim,
                &ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
                seed ^ 0x33,
            );
        });
    }

    fn mixed_ops() -> Vec<RamOp> {
        vec![
            RamOp::Write { word: 3, post: 5 },
            RamOp::Read { word: 3 },
            RamOp::None,
            RamOp::Write { word: 9, post: 4 },
            RamOp::Read { word: 9 },
            RamOp::Write { word: 3, post: 8 },
            RamOp::Read { word: 15 },
        ]
    }

    #[test]
    fn matches_reference_on_mixed_traffic() {
        run_parity(
            FixtureShape {
                log_t: 4,
                ram_k: 16,
            },
            mixed_ops(),
            31,
        );
    }

    #[test]
    fn matches_reference_on_odd_log_t() {
        // Five rounds: the lazy `ra` crosses its dense materialization and
        // the split LT collapses its lo tables mid-protocol.
        let mut ops = mixed_ops();
        ops.extend([
            RamOp::Write { word: 7, post: 2 },
            RamOp::Read { word: 7 },
            RamOp::None,
            RamOp::Read { word: 3 },
        ]);
        run_parity(
            FixtureShape {
                log_t: 5,
                ram_k: 16,
            },
            ops,
            37,
        );
    }

    #[test]
    fn matches_reference_on_single_round() {
        run_parity(
            FixtureShape { log_t: 1, ram_k: 8 },
            vec![RamOp::Write { word: 1, post: 4 }],
            41,
        );
    }
}
