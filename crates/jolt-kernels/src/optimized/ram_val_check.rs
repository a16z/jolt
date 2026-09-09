//! Optimized RAM value check (stage 4).
//!
//! Summand: `inc(j) · ra(j) · (LT(j, r_cycle) + γ)`.
//! [`LazyFoldedRa`] delays the dense `ra` table to `T/16`; [`SplitLt`] stores
//! `LT + γ` in ~√T space. [`IncColumn`] derives `inc` from raw trace values
//! for two rounds, then materializes at `T/4`.

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
use super::support::{
    bind_raw_twice, bound_pair, pin_derived_term, triple_product_round_evals, RoundProgress,
    SplitLt,
};
use super::OptimizedBackend;
use crate::reference::views::eq_table;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Remapped RAM address index, absent on no-access cycles.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct RamAddressIndices {
    addresses: Arc<Vec<u32>>,
}

impl ChunkIndexSource for RamAddressIndices {
    fn num_polys(&self) -> usize {
        1
    }

    fn cycles(&self) -> usize {
        self.addresses.len()
    }

    #[inline]
    fn index(&self, _i: usize, j: usize) -> Option<usize> {
        let address = self.addresses[j];
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

/// Builds the CPU kernel; the Metal route reuses it for its host fallback.
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

    // Reuse stage 2 addresses; collect only the short-lived values.
    let columns = RamAccessColumns::collect_full(session, witness, log_t)?;
    super::ram_trace::validate_addresses(&columns.addresses, 1usize << ram_log_k)?;
    let addresses = Arc::clone(&columns.addresses);

    Ok(RamValCheckKernel {
        progress: RoundProgress::new(log_t),
        inc: IncColumn::Raw(columns),
        ra: LazyFoldedRa::new(vec![eq_table(r_address)], RamAddressIndices { addresses }),
        lt: SplitLt::new_plus_constant(r_cycle, inputs.challenges.gamma),
    })
}

/// Raw trace values for two rounds; a dense `T/4` table afterward.
/// Reads have `post == pre`, and no-ops are zero, matching `RamInc`.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum IncColumn<F: JoltField> {
    Raw(RamAccessColumns),
    RawBound {
        columns: RamAccessColumns,
        #[cfg_attr(feature = "allocative", allocative(skip))]
        r1: F,
    },
    Bound(Polynomial<F>),
}

/// `inc(j)` from raw trace values.
#[inline]
fn raw_inc<F: JoltField>(columns: &RamAccessColumns, j: usize) -> F {
    F::from_i128(columns.post_values[j] as i128 - columns.pre_values[j] as i128)
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
pub(crate) struct RamValCheckKernel<F: JoltField> {
    progress: RoundProgress,
    inc: IncColumn<F>,
    ra: LazyFoldedRa<F, RamAddressIndices>,
    lt: SplitLt<F>,
}

impl<F: JoltField> RamValCheckKernel<F> {
    fn bind(&mut self, challenge: F) {
        let mut freed_columns = false;
        if let IncColumn::Bound(inc) = &mut self.inc {
            inc.bind_with_order(challenge, BindingOrder::LowToHigh);
        } else {
            let placeholder = IncColumn::Raw(RamAccessColumns {
                addresses: Arc::new(Vec::new()),
                pre_values: Vec::new(),
                post_values: Vec::new(),
            });
            self.inc = match std::mem::replace(&mut self.inc, placeholder) {
                // Round 1 reads bound pairs from the raw columns.
                IncColumn::Raw(columns) => IncColumn::RawBound {
                    columns,
                    r1: challenge,
                },
                // Materialize the second bind directly at `T/4`.
                IncColumn::RawBound { columns, r1 } => {
                    freed_columns = true;
                    IncColumn::Bound(bind_raw_twice(
                        |j| raw_inc(&columns, j),
                        columns.addresses.len(),
                        r1,
                        challenge,
                    ))
                }
                IncColumn::Bound(_) => unreachable!("bound inc binds in place above"),
            };
        }
        self.ra.bind(challenge);
        self.lt.bind(challenge);
        self.progress.advance();
        // Return freed `T`-sized columns before the stage boundary.
        if freed_columns {
            crate::mem::purge_retained_memory(self.progress.total());
        }
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

        // Each arm keeps representation branches outside the inner loop.
        let ra = |y: usize| self.ra.lo_hi(0, y);
        let lt = |y: usize| self.lt.pair(y);
        let evals = match &self.inc {
            IncColumn::Raw(columns) => triple_product_round_evals(
                columns.addresses.len() / 2,
                |y| (raw_inc(columns, 2 * y), raw_inc(columns, 2 * y + 1)),
                ra,
                lt,
            ),
            IncColumn::RawBound { columns, r1 } => triple_product_round_evals(
                columns.addresses.len() / 4,
                |y| {
                    let inc = |j| raw_inc(columns, j);
                    (bound_pair(inc, *r1, 2 * y), bound_pair(inc, *r1, 2 * y + 1))
                },
                ra,
                lt,
            ),
            IncColumn::Bound(inc) => {
                let inc = inc.evals();
                triple_product_round_evals(inc.len() / 2, |y| (inc[2 * y], inc[2 * y + 1]), ra, lt)
            }
        };

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
        // Advice outputs echo their input claims; they are not bound here.
        Ok(RamValCheckOutputClaims {
            untrusted_advice: inputs.untrusted_advice,
            trusted_advice: inputs.trusted_advice,
            program_image: inputs.program_image,
            ram_ra: self.ra.final_values()[0],
            ram_inc: match &self.inc {
                // Only when log_t = 0.
                IncColumn::Raw(columns) => raw_inc(columns, 0),
                // Only when log_t = 1.
                IncColumn::RawBound { columns, r1 } => bound_pair(|j| raw_inc(columns, j), *r1, 0),
                IncColumn::Bound(inc) => inc.evals()[0],
            },
        })
    }

    /// Check the bound split-LT value against the verifier's scalar path.
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
    use jolt_claims::protocols::jolt::geometry::ram::{
        ram_inc_val_check, ram_ra_val_check, RamValCheckInit,
    };
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

            // Independent input claim: `Σ inc · ra_folded · (LT + γ)`.
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
