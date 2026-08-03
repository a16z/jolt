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
//! - **Lazy `inc`** ([`IncColumn`]): round 0 derives
//!   `inc(j) = F::from_i128(post − pre)` straight from a kernel-local
//!   pre/post value view (one value-only trace walk; the address column is
//!   the session-shared Arc) — the same op the witness oracle applies per
//!   row, so the values are bit-identical — and round 1 serves the composed
//!   first-bind pairs `lo + r1·(hi − lo)` from the same view, so the dense
//!   bound column only materializes at `T/4` on the second bind. Neither
//!   the `T`- nor the `T/2`-sized field table ever exists; the raw columns
//!   (16 B/cycle, exactly a `T/2` field table's footprint) cover both early
//!   rounds.

use jolt_claims::protocols::jolt::{JoltDerivedId, RamValCheckPublic};
use jolt_field::{Field, OptimizedMul};
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage4::ram_val_check::{RamValCheck, RamValCheckOutputClaims};
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
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
struct RamAddressIndices {
    addresses: Arc<Vec<u64>>,
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

impl<F: Field> PrepareKernel<F, RamValCheck<F>> for OptimizedBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamValCheck<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamValCheck<F>>>, KernelError<F>> {
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

        // One value-only trace walk: the address column is the session-shared
        // Arc (parked at stage 2), so only pre/post are collected here — and
        // they die with the kernel's first cycle bind, not with the session.
        let columns = RamAccessColumns::collect_full(session, witness, log_t)?;
        super::ram_trace::validate_addresses(&columns.addresses, 1usize << ram_log_k)?;
        let addresses = Arc::clone(&columns.addresses);

        Ok(Box::new(RamValCheckKernel {
            progress: RoundProgress::new(log_t),
            inc: IncColumn::Raw(columns),
            ra: LazyFoldedRa::new(vec![eq_table(r_address)], RamAddressIndices { addresses }),
            lt: SplitLt::new_plus_constant(r_cycle, inputs.challenges.gamma),
        }))
    }
}

/// The `inc` column in its round-dependent representation. Round 0 serves
/// `inc(j)` straight from the kernel-local access columns, round 1 the
/// composed first-bind pairs from the same columns; the second bind
/// materializes the dense bound column at `T/4` and frees them.
///
/// The derivation is exact for every row kind: the witness oracle's `RamInc`
/// is `F::from_i128(post − pre)` for writes and `F::from_i128(0)` otherwise,
/// and the columns carry `post == pre` on reads and zeros on no-ops, so
/// [`raw_inc`] reproduces the oracle's field values bit-for-bit.
enum IncColumn<F: Field> {
    Raw(RamAccessColumns),
    RawBound { columns: RamAccessColumns, r1: F },
    Bound(Polynomial<F>),
}

/// `inc(j)` from the raw columns — the witness oracle's exact op.
#[inline]
fn raw_inc<F: Field>(columns: &RamAccessColumns, j: usize) -> F {
    F::from_i128(columns.post_values[j] as i128 - columns.pre_values[j] as i128)
}

/// The composed first-bind value `inc(y)` on the half domain: the exact
/// `lo + r·(hi − lo)` op the eager `T/2` materialization applied.
#[inline]
fn raw_bound_inc<F: Field>(columns: &RamAccessColumns, r1: F, y: usize) -> F {
    let lo = raw_inc::<F>(columns, 2 * y);
    let hi = raw_inc::<F>(columns, 2 * y + 1);
    lo + r1.mul_0_optimized(hi - lo)
}

struct RamValCheckKernel<F: Field> {
    progress: RoundProgress,
    inc: IncColumn<F>,
    ra: LazyFoldedRa<F, RamAddressIndices>,
    lt: SplitLt<F>,
}

#[cfg(feature = "allocative")]
crate::optimized::impl_field_allocative!(RamValCheckKernel, |kernel| {
    // Raw inc owns the kernel-local pre/post value columns; the shared
    // address column is attributed once, through the ra source below.
    let inc = match &kernel.inc {
        IncColumn::Raw(columns) | IncColumn::RawBound { columns, .. } => {
            crate::backend::vec_heap_bytes(&columns.pre_values)
                + crate::backend::vec_heap_bytes(&columns.post_values)
        }
        IncColumn::Bound(inc) => crate::backend::poly_heap_bytes(inc),
    };
    inc + kernel
        .ra
        .heap_bytes(|source| crate::backend::arc_vec_heap_bytes(&source.addresses))
        + kernel.lt.heap_bytes()
});

impl<F: Field> RamValCheckKernel<F> {
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
                // The first bind is free: round 1 serves the composed
                // pairs straight from the raw columns.
                IncColumn::Raw(columns) => IncColumn::RawBound {
                    columns,
                    r1: challenge,
                },
                // Materialize the second bind directly at `T/4`, each
                // level the same pair op `lo + r·(hi − lo)` as
                // `Polynomial::bind_with_order`.
                IncColumn::RawBound { columns, r1 } => {
                    let quarter = columns.addresses.len() / 4;
                    let pair = |z: usize| {
                        let lo = raw_bound_inc::<F>(&columns, r1, 2 * z);
                        let hi = raw_bound_inc::<F>(&columns, r1, 2 * z + 1);
                        lo + challenge.mul_0_optimized(hi - lo)
                    };
                    #[cfg(feature = "parallel")]
                    let bound: Vec<F> = (0..quarter).into_par_iter().map(pair).collect();
                    #[cfg(not(feature = "parallel"))]
                    let bound: Vec<F> = (0..quarter).map(pair).collect();
                    freed_columns = true;
                    IncColumn::Bound(Polynomial::new(bound))
                }
                IncColumn::Bound(_) => unreachable!("bound inc binds in place above"),
            };
        }
        self.ra.bind(challenge);
        self.lt.bind(challenge);
        self.progress.advance();
        // The second bind frees the T-sized pre/post value columns; this
        // member is tail-aligned, so that happens rounds after the sibling
        // register kernel's transition purges — the dead pages would sit in
        // the allocator's freed-large-block cache until the stage boundary.
        if freed_columns {
            crate::mem::purge_staging(self.progress.total());
        }
    }
}

impl<F: Field> ProveRounds<F> for RamValCheckKernel<F> {
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

        // Monomorphized per inc source: no per-element representation branch
        // in the accumulate loop (each arm instantiates the shared sampler
        // with its own inc closure).
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
                    (
                        raw_bound_inc(columns, *r1, 2 * y),
                        raw_bound_inc(columns, *r1, 2 * y + 1),
                    )
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

impl<F: Field> SumcheckKernel<F> for RamValCheckKernel<F> {
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
            ram_inc: match &self.inc {
                // Raw only when log_t = 0 (no cycle rounds at all).
                IncColumn::Raw(columns) => raw_inc(columns, 0),
                // RawBound only when log_t = 1 (one composed pair).
                IncColumn::RawBound { columns, r1 } => raw_bound_inc(columns, *r1, 0),
                IncColumn::Bound(inc) => inc.evals()[0],
            },
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
        pin_derived_term(
            relation,
            JoltDerivedId::from(RamValCheckPublic::LtCyclePlusGamma),
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
    use jolt_field::{Fr, FromPrimitiveInt};
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
