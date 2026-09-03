//! The optimized registers value-evaluation (stage 5) kernel.
//!
//! Byte-parity contract: identical round polynomials and output claims to the
//! reference kernel (`reference/registers_val_evaluation.rs`), which folds the
//! `(2^7 × T)` one-hot `rd_wa` grid into a dense cycle table at prepare time
//! and binds three dense `T`-sized tables per round.
//!
//! Techniques ported from
//! `jolt-prover-legacy/src/zkvm/registers/val_evaluation.rs`:
//!
//! - **Lazy one-hot `wa`** (legacy `RaPolynomial`, first-round form): round 0
//!   serves `wa(j) = eq(r_address)[rd_j]` straight from the per-cycle hot
//!   indices and the K-sized eq table — the `K × T` grid is never
//!   materialized, and the reference's prepare-time `address_fold` over it is
//!   gone. The dense bound vector (T/2) appears only at the first bind.
//! - **Rows-until-first-bind increments**: round 0 reads `RdInc` from typed
//!   rows; the first bind creates a dense `T/2` table.
//! - **Split LT** ([`SplitLt`], legacy `LtPolynomial`): `LT(j, r_cycle)` is
//!   served from three ~√T tables instead of a dense `T`-sized one.
//! - **Eval-at-{0,2,3} sampling** with the engine hint supplying s(1)
//!   (legacy samples {1,2,∞}; both interpolate the same cubic exactly).
//! - **Deferred-reduction accumulation** of the triple products.
//!
//! The per-cycle `rd` indices are reclaimed from the proof session when the
//! stage-4 optimized kernel parked them, avoiding a second trace walk.

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::geometry::registers::rd_inc_val_evaluation;
use jolt_claims::protocols::jolt::{JoltDerivedId, RegistersValEvaluationPublic};
use jolt_field::{Accumulator, JoltField};
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage5::registers_val_evaluation::{
    RegistersValEvaluation, RegistersValEvaluationOutputClaims,
};
use jolt_witness::witnesses::{RdInc, ToField};
use jolt_witness::{JoltWitnessPlane, WitnessBundle};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::registers_read_write::{RegisterCycleRow, SharedRdIndices};
use super::support::{
    collect_rows, pin_derived_term, triple_product_round_evals, BundleStore, RoundProgress, SplitLt,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Address indices before the first bind; a dense table afterward.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum WaState<F: JoltField> {
    Indices {
        rd: Vec<Option<u8>>,
        eq_address: Vec<F>,
    },
    Dense(Polynomial<F>),
}

impl<F: JoltField> WaState<F> {
    #[inline]
    fn pair(&self, y: usize) -> (F, F) {
        match self {
            Self::Indices { rd, eq_address } => {
                let value = |j: usize| rd[j].map_or(F::zero(), |k| eq_address[k as usize]);
                (value(2 * y), value(2 * y + 1))
            }
            Self::Dense(table) => {
                let evals = table.evals();
                (evals[2 * y], evals[2 * y + 1])
            }
        }
    }

    fn bind(&mut self, r: F) {
        match self {
            Self::Indices { rd, eq_address } => {
                let value = |j: usize| rd[j].map_or(F::zero(), |k| eq_address[k as usize]);
                let half = rd.len() / 2;
                let bind_pair = |y: usize| {
                    let lo = value(2 * y);
                    lo + r * (value(2 * y + 1) - lo)
                };
                #[cfg(feature = "parallel")]
                let dense: Vec<F> = (0..half).into_par_iter().map(bind_pair).collect();
                #[cfg(not(feature = "parallel"))]
                let dense: Vec<F> = (0..half).map(bind_pair).collect();
                *self = Self::Dense(Polynomial::new(dense));
            }
            Self::Dense(table) => table.bind_with_order(r, BindingOrder::LowToHigh),
        }
    }

    fn final_value(&self) -> F {
        match self {
            Self::Dense(table) => {
                debug_assert_eq!(table.len(), 1);
                table.evals()[0]
            }
            Self::Indices { .. } => unreachable!("bound at least once before extraction"),
        }
    }
}

pub struct OptimizedRegistersValEvaluation;

impl<F: JoltField> PrepareKernel<F, RegistersValEvaluation<F>> for OptimizedRegistersValEvaluation {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RegistersValEvaluation<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersValEvaluation<F>>>, KernelError<F>>
    {
        let log_t = inputs.relation.trace_dimensions().log_t();
        if log_t == 0 {
            return Err(KernelError::Unsupported {
                reason: "optimized registers val-evaluation requires at least one cycle round",
            });
        }
        let registers_val_point: &[F] = &inputs.points.registers_val;
        if registers_val_point.len() != REGISTER_ADDRESS_BITS + log_t {
            return Err(KernelError::InvariantViolation {
                reason: "registers value-evaluation input point has the wrong variable count",
            });
        }
        let (r_address, r_cycle) = registers_val_point.split_at(REGISTER_ADDRESS_BITS);
        let cycles = 1usize << log_t;

        let rows = witness
            .shape(rd_inc_val_evaluation().polynomial_id())?
            .rows();
        if rows != cycles {
            return Err(KernelError::TableSizeMismatch {
                table: format!("{:?}", rd_inc_val_evaluation()),
                expected: cycles,
                got: rows,
            });
        }
        let inc = IncState::Rows(BundleStore::resolve(witness, cycles)?);

        // Reclaim the rd hot indices the stage-4 kernel parked; collect them
        // from the row source otherwise (reference-only stage 4, tests).
        let rd = match session.take::<SharedRdIndices>() {
            Some(SharedRdIndices(rd)) if rd.len() == cycles => rd,
            _ => collect_rows::<RegisterCycleRow>(witness, cycles)?
                .iter()
                .map(|row| row.rd.map(|(k, ..)| k))
                .collect(),
        };

        Ok(Box::new(ValEvaluationKernel {
            progress: RoundProgress::new(log_t),
            inc,
            wa: WaState::Indices {
                rd,
                eq_address: EqPolynomial::<F>::evals(r_address, None),
            },
            lt: SplitLt::new(r_cycle),
        }))
    }
}

/// Trace rows before the first bind; a dense table afterward.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum IncState<F: JoltField> {
    Rows(BundleStore<RdIncRow>),
    Dense(Polynomial<F>),
}

/// The single-column bundle behind the increment table.
#[derive(Clone, Copy, Debug, WitnessBundle)]
struct RdIncRow {
    rd_inc: RdInc,
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct ValEvaluationKernel<F: JoltField> {
    progress: RoundProgress,
    inc: IncState<F>,
    wa: WaState<F>,
    lt: SplitLt<F>,
}

fn row_unavailable<F: JoltField>() -> SumcheckError<F> {
    SumcheckError::MissingEvaluationSource {
        kind: "registers increment trace rows",
    }
}

impl<F: JoltField> ValEvaluationKernel<F> {
    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        match &mut self.inc {
            IncState::Dense(inc) => inc.bind_with_order(challenge, BindingOrder::LowToHigh),
            IncState::Rows(store) => {
                // Bind row pairs directly into a half-length field table.
                debug_assert_eq!(self.progress.bound(), 0);
                let half = (1usize << self.progress.total()) / 2;
                let access = store.access();
                let bound = |y: usize| -> Result<F, SumcheckError<F>> {
                    let even: RdIncRow = access.row(2 * y).map_err(|_| row_unavailable())?;
                    let odd: RdIncRow = access.row(2 * y + 1).map_err(|_| row_unavailable())?;
                    let (lo, hi) = (even.rd_inc.to_field::<F>(), odd.rd_inc.to_field::<F>());
                    Ok(lo + challenge * (hi - lo))
                };
                #[cfg(feature = "parallel")]
                let table = (0..half)
                    .into_par_iter()
                    .map(bound)
                    .collect::<Result<Vec<F>, _>>()?;
                #[cfg(not(feature = "parallel"))]
                let table = (0..half).map(bound).collect::<Result<Vec<F>, _>>()?;
                self.inc = IncState::Dense(Polynomial::new(table));
            }
        }
        self.wa.bind(challenge);
        self.lt.bind(challenge);
        self.progress.advance();
        Ok(())
    }
}

impl<F: JoltField> ProveRounds<F> for ValEvaluationKernel<F> {
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
            self.bind(challenge)?;
        }

        // Evaluate at 0, 2, 3; the engine supplies s(1).
        // Round 0 reads rows fallibly; later rounds read the dense table.
        let evals = match &self.inc {
            IncState::Rows(store) => {
                debug_assert_eq!(self.progress.bound(), 0);
                let half = (1usize << self.progress.total()) / 2;
                let access = store.access();
                let group = |y: usize,
                             acc: &mut [F::Accumulator; 3]|
                 -> Result<(), SumcheckError<F>> {
                    let even: RdIncRow = access.row(2 * y).map_err(|_| row_unavailable())?;
                    let odd: RdIncRow = access.row(2 * y + 1).map_err(|_| row_unavailable())?;
                    let (inc_0, inc_1) = (even.rd_inc.to_field::<F>(), odd.rd_inc.to_field::<F>());
                    let (wa_0, wa_1) = self.wa.pair(y);
                    let (lt_0, lt_1) = self.lt.pair(y);
                    let (inc_m, wa_m, lt_m) = (inc_1 - inc_0, wa_1 - wa_0, lt_1 - lt_0);
                    let (inc_2, wa_2, lt_2) = (inc_1 + inc_m, wa_1 + wa_m, lt_1 + lt_m);
                    acc[0].fmadd(inc_0 * wa_0, lt_0);
                    acc[1].fmadd(inc_2 * wa_2, lt_2);
                    acc[2].fmadd((inc_2 + inc_m) * (wa_2 + wa_m), lt_2 + lt_m);
                    Ok(())
                };
                #[cfg(feature = "parallel")]
                let evals = (0..half)
                    .into_par_iter()
                    .try_fold(
                        || [F::Accumulator::default(); 3],
                        |mut acc, y| {
                            group(y, &mut acc)?;
                            Ok(acc)
                        },
                    )
                    .map(|acc| acc.map(|acc| acc.map(F::Accumulator::reduce)))
                    .try_reduce(
                        || [F::zero(); 3],
                        |a, b| Ok([a[0] + b[0], a[1] + b[1], a[2] + b[2]]),
                    )?;
                #[cfg(not(feature = "parallel"))]
                let evals = {
                    let mut acc = [F::Accumulator::default(); 3];
                    for y in 0..half {
                        group(y, &mut acc)?;
                    }
                    acc.map(F::Accumulator::reduce)
                };
                evals
            }
            IncState::Dense(inc_poly) => {
                let inc = inc_poly.evals();
                triple_product_round_evals(
                    inc_poly.len() / 2,
                    |y| (inc[2 * y], inc[2 * y + 1]),
                    |y| self.wa.pair(y),
                    |y| self.lt.pair(y),
                )
            }
        };

        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: JoltField> SumcheckKernel<F> for ValEvaluationKernel<F> {
    type Relation = RegistersValEvaluation<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<RegistersValEvaluationOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let IncState::Dense(inc) = &self.inc else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "increment table absent after full binding",
            });
        };
        Ok(RegistersValEvaluationOutputClaims {
            rd_inc: inc.evals()[0],
            rd_wa: self.wa.final_value(),
        })
    }

    /// Pin the split-LT tables to the verifier's scalar path: the fully bound
    /// LT value must equal `derive_output_term(LtCycle)`.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let id = JoltDerivedId::from(RegistersValEvaluationPublic::LtCycle);
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
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::{
        TraceDimensions, REGISTER_ADDRESS_BITS,
    };
    use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
    use jolt_claims::NoChallenges;
    use jolt_field::Fr;
    use jolt_poly::Polynomial;
    use jolt_verifier::stages::stage5::registers_val_evaluation::{
        RegistersValEvaluation, RegistersValEvaluationInputClaims,
    };
    use jolt_witness::{collect_bundles, JoltWitnessOracle};

    use super::super::registers_read_write::test_support::{
        assert_kernel_parity_with_session, assert_nontrivial, challenge_sequence,
        structured_fixture, TraceFixture,
    };
    use super::super::registers_read_write::{RegisterCycleRow, SharedRdIndices};
    use super::OptimizedRegistersValEvaluation;
    use crate::ProofSession;

    /// How the optimized kernel sources its per-cycle rd indices.
    enum IndexSource {
        /// Collected from the row source inside `prepare`.
        Collect,
        /// Reclaimed from a session carry parked by stage 4.
        Parked,
        /// A stale (wrong-length) carry is parked; `prepare` must fall back
        /// to collecting.
        StaleParked,
    }

    fn run_parity(fixture: TraceFixture, log_t: usize, seed: u64, source: &IndexSource) {
        fixture.with_plane(log_t, |backend| {
            let relation = RegistersValEvaluation::<Fr>::new(TraceDimensions::new(log_t));
            let point = challenge_sequence(REGISTER_ADDRESS_BITS + log_t, seed ^ 0x3C3C);
            let grid = JoltWitnessOracle::<Fr>::oracle_table(
                backend,
                JoltPolynomialId::Virtual(JoltVirtualPolynomial::RegistersVal),
            )
            .unwrap();
            let input_claim = Polynomial::new(grid).evaluate(&point);
            assert_nontrivial(input_claim);

            let mut session = ProofSession::default();
            match source {
                IndexSource::Collect => {}
                IndexSource::Parked => {
                    let rows: Vec<RegisterCycleRow> = collect_bundles(backend, 1 << log_t).unwrap();
                    session.park(SharedRdIndices(
                        rows.iter().map(|row| row.rd.map(|(k, ..)| k)).collect(),
                    ));
                }
                IndexSource::StaleParked => {
                    session.park(SharedRdIndices(vec![None; (1 << log_t) + 1]));
                }
            }

            let claims = RegistersValEvaluationInputClaims {
                registers_val: input_claim,
            };
            let points = RegistersValEvaluationInputClaims {
                registers_val: point,
            };
            let round_challenges = challenge_sequence(log_t, seed);
            assert_kernel_parity_with_session(
                &mut session,
                &OptimizedRegistersValEvaluation,
                backend,
                &relation,
                &claims,
                &points,
                &NoChallenges::default(),
                input_claim,
                &round_challenges,
            );
        });
    }

    #[test]
    fn parity_structured_odd_log_t() {
        run_parity(structured_fixture(8), 3, 53, &IndexSource::Collect);
    }

    #[test]
    fn parity_structured_even_log_t() {
        run_parity(structured_fixture(16), 4, 59, &IndexSource::Collect);
    }

    #[test]
    fn parity_minimal_padded_trace() {
        let mut fixture = TraceFixture::new();
        fixture.op(Some(4), Some(1), None);
        fixture.op(Some(4), Some(4), Some(4));
        fixture.op(None, Some(4), None);
        run_parity(fixture, 2, 61, &IndexSource::Collect);
    }

    #[test]
    fn parity_single_cycle_round() {
        let mut fixture = TraceFixture::new();
        fixture.op(Some(11), Some(11), None);
        fixture.op(Some(11), None, Some(11));
        run_parity(fixture, 1, 67, &IndexSource::Collect);
    }

    #[test]
    fn parity_with_parked_indices() {
        run_parity(structured_fixture(8), 3, 71, &IndexSource::Parked);
    }

    #[test]
    fn parity_with_stale_parked_indices_falls_back() {
        run_parity(structured_fixture(8), 3, 73, &IndexSource::StaleParked);
    }
}
