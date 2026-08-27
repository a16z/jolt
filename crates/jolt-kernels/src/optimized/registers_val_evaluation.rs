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
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage5::registers_val_evaluation::{
    RegistersValEvaluation, RegistersValEvaluationOutputClaims,
};
use jolt_witness::witnesses::{RdInc, ToField};
use jolt_witness::{JoltWitnessPlane, RandomAccessRows, WitnessBundle};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::registers_read_write::{RegisterCycleRow, SharedRdIndices};
use super::support::{
    bind_pairs, collect_par_map, collect_rows, pin_derived_term, triple_product_round_evals,
    RoundProgress, SplitLt,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// The write-address column: hot indices plus the address eq table until the
/// first bind, a dense bound vector afterwards. The `K × T` grid never exists.
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F")
)]
pub(crate) enum WaState<F> {
    Indices {
        rd: Vec<Option<u8>>,
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
        eq_address: Vec<F>,
    },
    Dense(#[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))] Vec<F>),
}

impl<F: JoltField> WaState<F> {
    #[inline]
    fn pair(&self, y: usize) -> (F, F) {
        match self {
            Self::Indices { rd, eq_address } => {
                let value = |j: usize| rd[j].map_or(F::zero(), |k| eq_address[k as usize]);
                (value(2 * y), value(2 * y + 1))
            }
            Self::Dense(table) => (table[2 * y], table[2 * y + 1]),
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
                *self = Self::Dense(dense);
            }
            Self::Dense(table) => bind_pairs(table, r),
        }
    }

    fn final_value(&self) -> F {
        match self {
            Self::Dense(table) => {
                debug_assert_eq!(table.len(), 1);
                table[0]
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
        let parts = ValEvaluationParts::collect(session, witness, &inputs)?;
        Ok(Box::new(ValEvaluationKernel::from_parts(parts)))
    }
}

/// The kernel's prepared tables, shared with the Metal slot (which owns its
/// device buffers but reuses this collection and falls back to
/// [`ValEvaluationKernel`] on any device failure).
pub(crate) struct ValEvaluationParts<F: JoltField> {
    pub(crate) log_t: usize,
    pub(crate) inc: IncSource<F>,
    pub(crate) rd: Vec<Option<u8>>,
    pub(crate) eq_address: Vec<F>,
    pub(crate) lt: SplitLt<F>,
}

impl<F: JoltField> ValEvaluationParts<F> {
    pub(crate) fn collect(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: &ProverInputs<'_, F, RegistersValEvaluation<F>>,
    ) -> Result<Self, KernelError<F>> {
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

        let inc = IncSource::collect(witness, cycles)?;

        // Reclaim the rd hot indices the stage-4 kernel parked; collect them
        // from the row source otherwise (reference-only stage 4, tests).
        let rd = match session.take::<SharedRdIndices>() {
            Some(SharedRdIndices(rd)) if rd.len() == cycles => rd,
            _ => collect_rows::<RegisterCycleRow>(witness, cycles)?
                .iter()
                .map(|row| row.rd.map(|(k, ..)| k))
                .collect(),
        };

        Ok(Self {
            log_t,
            inc,
            rd,
            eq_address: EqPolynomial::<F>::evals(r_address, None),
            lt: SplitLt::new(r_cycle),
        })
    }
}

/// The increment table's lifecycle: deferred to the member's first active
/// round on slice-backed sources, dense from prepare otherwise.
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
pub(crate) enum IncSource<F: JoltField> {
    /// The witness plane owns these rows; it reports them itself.
    #[cfg_attr(feature = "allocative", allocative(skip))]
    Deferred(RandomAccessRows),
    Ready(Polynomial<F>),
}

impl<F: JoltField> IncSource<F> {
    /// Slice-backed sources defer the dense increment table to the member's
    /// first active round: the member is tail-aligned in the stage-5 batch,
    /// so a prepare-time table would co-inhabit the prover's peak moment (the
    /// instruction kernel's address/cycle handoff) doing nothing. Values are
    /// identical either way — the same extractor over the same rows.
    pub(crate) fn collect(
        witness: &dyn JoltWitnessPlane<F>,
        cycles: usize,
    ) -> Result<Self, KernelError<F>> {
        Ok(match witness.random_access() {
            Some(rows) if rows.cycles() == cycles => Self::Deferred(rows),
            _ => {
                let inc_table: Vec<F> =
                    witness.oracle_table(rd_inc_val_evaluation().polynomial_id())?;
                if inc_table.len() != cycles {
                    return Err(KernelError::TableSizeMismatch {
                        table: format!("{:?}", rd_inc_val_evaluation()),
                        expected: cycles,
                        got: inc_table.len(),
                    });
                }
                Self::Ready(Polynomial::new(inc_table))
            }
        })
    }

    /// Materialize the deferred increment table; a no-op once ready.
    fn ensure(&mut self) -> Result<(), SumcheckError<F>> {
        if let Self::Deferred(owned) = &*self {
            let table: Vec<F> =
                collect_par_map(owned, owned.cycles(), |row: RdIncRow| row.rd_inc.to_field())
                    .map_err(|_| SumcheckError::MissingEvaluationSource {
                        kind: "deferred registers increment table",
                    })?;
            *self = Self::Ready(Polynomial::new(table));
        }
        Ok(())
    }

    /// The dense table for device upload, materializing a deferred source and
    /// leaving an empty slot behind (the Metal build re-collects on failure).
    #[cfg(feature = "metal")]
    pub(crate) fn take_table(&mut self) -> Result<Vec<F>, SumcheckError<F>> {
        self.ensure()?;
        match self {
            Self::Ready(poly) => {
                Ok(std::mem::replace(poly, Polynomial::new(Vec::new())).into_evals())
            }
            Self::Deferred(_) => Err(SumcheckError::MissingEvaluationSource {
                kind: "deferred registers increment table",
            }),
        }
    }

    /// True once [`Self::take_table`] consumed the table.
    #[cfg(feature = "metal")]
    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::Ready(poly) => poly.len() == 0,
            Self::Deferred(_) => false,
        }
    }
}

/// The single-column bundle behind the deferred increment table.
#[derive(Clone, Copy, Debug, WitnessBundle)]
struct RdIncRow {
    rd_inc: RdInc,
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
pub(crate) struct ValEvaluationKernel<F: JoltField> {
    progress: RoundProgress,
    inc: IncSource<F>,
    wa: WaState<F>,
    lt: SplitLt<F>,
}

impl<F: JoltField> ValEvaluationKernel<F> {
    pub(crate) fn from_parts(parts: ValEvaluationParts<F>) -> Self {
        Self {
            progress: RoundProgress::new(parts.log_t),
            inc: parts.inc,
            wa: WaState::Indices {
                rd: parts.rd,
                eq_address: parts.eq_address,
            },
            lt: parts.lt,
        }
    }

    /// Resume mid-sumcheck from a device kernel's live state (unified-memory
    /// slices copied out): `rounds_bound` challenges already folded into
    /// every table.
    #[cfg(feature = "metal")]
    pub(crate) fn from_bound_state(
        rounds: usize,
        inc: Vec<F>,
        wa: WaState<F>,
        lt: SplitLt<F>,
        rounds_bound: usize,
    ) -> Self {
        let mut progress = RoundProgress::new(rounds);
        for _ in 0..rounds_bound {
            progress.advance();
        }
        Self {
            progress,
            inc: IncSource::Ready(Polynomial::new(inc)),
            wa,
            lt,
        }
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
        self.inc.ensure()?;
        if let Some(challenge) = bind {
            if let IncSource::Ready(inc) = &mut self.inc {
                inc.bind_with_order(challenge, BindingOrder::LowToHigh);
            }
            self.wa.bind(challenge);
            self.lt.bind(challenge);
            self.progress.advance();
        }

        let IncSource::Ready(inc_poly) = &self.inc else {
            return Err(SumcheckError::MissingEvaluationSource {
                kind: "registers increment table",
            });
        };
        let half = inc_poly.len() / 2;
        let inc = inc_poly.evals();
        let evals = triple_product_round_evals(
            half,
            |y| (inc[2 * y], inc[2 * y + 1]),
            |y| self.wa.pair(y),
            |y| self.lt.pair(y),
        );

        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.inc.ensure()?;
        if let IncSource::Ready(inc) = &mut self.inc {
            inc.bind_with_order(bind, BindingOrder::LowToHigh);
        }
        self.wa.bind(bind);
        self.lt.bind(bind);
        self.progress.advance();
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for ValEvaluationKernel<F> {
    type Relation = RegistersValEvaluation<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<RegistersValEvaluationOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let IncSource::Ready(inc) = &self.inc else {
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
