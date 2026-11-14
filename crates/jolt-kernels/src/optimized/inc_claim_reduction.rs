//! Optimized increment claim reduction (stage 6b).
//!
//! The four eq leaves collapse into two tables:
//!
//! `A(j) = eq(r_ram_rw, j) + γ·eq(r_ram_val, j)`
//! `B(j) = γ²·eq(s_reg_rw, j) + γ³·eq(s_reg_val, j)`
//!
//! The summand is `A·RamInc + B·RdInc`. [`PairedEq`] stores `A` and `B` in
//! split tables. Increment columns stay in trace rows until the first bind,
//! avoiding four `T`-length eq tables and two `T`-length increment tables.

use jolt_claims::protocols::jolt::geometry::claim_reductions::increments::{
    ram_inc_reduced, rd_inc_reduced,
};
use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_field::JoltField;
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_verifier::stages::relations::{ConcreteSumcheck, SumcheckInputClaims};
use jolt_verifier::stages::stage6b::inc_claim_reduction::{
    IncClaimReduction, IncClaimReductionOutputClaims,
};
use jolt_witness::witnesses::{RamInc, RdInc, ToField};
use jolt_witness::{JoltWitnessPlane, WitnessBundle};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use super::support::merge_evals;
use super::support::{
    bind_all, eq_table, pair, par_sum_pair_groups, round_poly_from_skipped_evals, scaled_eq_table,
    BundleStore, RoundProgress,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

pub struct OptimizedIncClaimReduction;

#[derive(Clone, Copy, Debug, WitnessBundle)]
struct IncRow {
    #[opening(committed = RamInc)]
    ram_inc: RamInc,
    #[opening(committed = RdInc)]
    rd_inc: RdInc,
}

impl<F: JoltField> PrepareKernel<F, IncClaimReduction<F>> for OptimizedIncClaimReduction {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, IncClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = IncClaimReduction<F>>>, KernelError<F>> {
        Ok(Box::new(IncKernel::new(witness, inputs)?))
    }
}

impl<F: JoltField> IncKernel<F> {
    fn new(
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, IncClaimReduction<F>>,
    ) -> Result<Self, KernelError<F>> {
        let relation = inputs.relation;
        let cycle_points = relation.cycle_points();
        for point in cycle_points {
            if point.len() != relation.rounds() {
                return Err(KernelError::InvariantViolation {
                    reason: "increment reduction cycle point has the wrong variable count",
                });
            }
        }
        let cycles = 1usize << relation.rounds();
        for id in [ram_inc_reduced(), rd_inc_reduced()] {
            let rows = witness.shape(id.polynomial_id())?.rows();
            if rows != cycles {
                return Err(KernelError::TableSizeMismatch {
                    table: format!("{id:?}"),
                    expected: cycles,
                    got: rows,
                });
            }
        }

        let gamma = inputs.challenges.gamma;
        let gamma_squared = gamma * gamma;
        let ram_weights = PairedEq::new(cycle_points[0], F::one(), cycle_points[1], gamma);
        let rd_weights = PairedEq::new(
            cycle_points[2],
            gamma_squared,
            cycle_points[3],
            gamma_squared * gamma,
        );

        let incs = if relation.rounds() == 0 {
            // No bind occurs on a single-cycle domain.
            let dense = |id: JoltOpeningId| witness.oracle_table(id.polynomial_id());
            IncState::Dense {
                ram: Polynomial::new(dense(ram_inc_reduced())?),
                rd: Polynomial::new(dense(rd_inc_reduced())?),
            }
        } else {
            IncState::Rows(BundleStore::resolve(witness, cycles)?)
        };

        Ok(Self {
            progress: RoundProgress::new(relation.rounds()),
            incs,
            ram_weights,
            rd_weights,
        })
    }
}

/// `s₁·eq(p₁, ·) + s₂·eq(p₂, ·)` in four ~√T split tables.
/// Binding folds the exhausted low scalars into one dense high table.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) enum PairedEq<F> {
    Split {
        lo1: Vec<F>,
        hi1: Vec<F>,
        lo2: Vec<F>,
        hi2: Vec<F>,
    },
    Dense(Vec<F>),
}

impl<F: JoltField> PairedEq<F> {
    pub(crate) fn new(p1: &[F], s1: F, p2: &[F], s2: F) -> Self {
        debug_assert_eq!(p1.len(), p2.len());
        let mid = p1.len() / 2;
        if mid == 0 {
            let mut table = scaled_eq_table(p1, s1);
            for (acc, term) in table.iter_mut().zip(scaled_eq_table(p2, s2)) {
                *acc += term;
            }
            return Self::Dense(table);
        }
        let (p1_hi, p1_lo) = p1.split_at(p1.len() - mid);
        let (p2_hi, p2_lo) = p2.split_at(p2.len() - mid);
        Self::Split {
            lo1: eq_table(p1_lo),
            hi1: scaled_eq_table(p1_hi, s1),
            lo2: eq_table(p2_lo),
            hi2: scaled_eq_table(p2_hi, s2),
        }
    }

    /// The combined table's `(lo, hi)` sumcheck pair at group `y` under
    /// low-to-high pairing.
    #[inline]
    pub(crate) fn pair(&self, y: usize) -> (F, F) {
        match self {
            Self::Split { lo1, hi1, lo2, hi2 } => {
                let lo_len = lo1.len();
                let j = 2 * y;
                let hi = j / lo_len;
                debug_assert!(lo_len >= 2, "adjacent lo indices share the hi part");
                let (a, b) = (hi1[hi], hi2[hi]);
                (
                    a * lo1[j % lo_len] + b * lo2[j % lo_len],
                    a * lo1[(j + 1) % lo_len] + b * lo2[(j + 1) % lo_len],
                )
            }
            Self::Dense(table) => (table[2 * y], table[2 * y + 1]),
        }
    }

    pub(crate) fn bind(&mut self, r: F) {
        match self {
            Self::Split { lo1, hi1, lo2, hi2 } => {
                let half = lo1.len() / 2;
                for lo in [&mut *lo1, &mut *lo2] {
                    for y in 0..half {
                        let even = lo[2 * y];
                        lo[y] = even + r * (lo[2 * y + 1] - even);
                    }
                    lo.truncate(half);
                }
                if half == 1 {
                    let (s1, s2) = (lo1[0], lo2[0]);
                    let dense: Vec<F> = hi1
                        .iter()
                        .zip(hi2.iter())
                        .map(|(&a, &b)| a * s1 + b * s2)
                        .collect();
                    *self = Self::Dense(dense);
                }
            }
            Self::Dense(table) => {
                let half = table.len() / 2;
                for y in 0..half {
                    let even = table[2 * y];
                    table[y] = even + r * (table[2 * y + 1] - even);
                }
                table.truncate(half);
            }
        }
    }
    #[cfg(feature = "field-inline")]
    pub(crate) fn final_value(&self) -> Option<F> {
        match self {
            Self::Dense(table) if table.len() == 1 => table.first().copied(),
            _ => None,
        }
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum IncState<F: JoltField> {
    Rows(BundleStore<IncRow>),
    Dense {
        ram: Polynomial<F>,
        rd: Polynomial<F>,
    },
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct IncKernel<F: JoltField> {
    progress: RoundProgress,
    incs: IncState<F>,
    ram_weights: PairedEq<F>,
    rd_weights: PairedEq<F>,
}

fn row_unavailable<F: JoltField>() -> SumcheckError<F> {
    SumcheckError::MissingEvaluationSource {
        kind: "increment claim-reduction trace rows",
    }
}

impl<F: JoltField> IncKernel<F> {
    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        if let IncState::Dense { ram, rd } = &mut self.incs {
            bind_all([ram, rd], challenge);
        } else {
            let (ram, rd) = self.materialize_bound(challenge)?;
            self.incs = IncState::Dense { ram, rd };
        }
        self.ram_weights.bind(challenge);
        self.rd_weights.bind(challenge);
        self.progress.advance();
        Ok(())
    }

    fn materialize_bound(
        &self,
        challenge: F,
    ) -> Result<(Polynomial<F>, Polynomial<F>), SumcheckError<F>> {
        let IncState::Rows(store) = &self.incs else {
            unreachable!("materialize_bound is only called in the rows state");
        };
        debug_assert_eq!(self.progress.bound(), 0);
        let half = (1usize << self.progress.total()) / 2;
        let access = store.access();
        let bound = |y: usize| -> Result<(F, F), SumcheckError<F>> {
            let even: IncRow = access.row(2 * y).map_err(|_| row_unavailable())?;
            let odd: IncRow = access.row(2 * y + 1).map_err(|_| row_unavailable())?;
            let (ram_even, ram_odd) = (even.ram_inc.to_field::<F>(), odd.ram_inc.to_field::<F>());
            let (rd_even, rd_odd) = (even.rd_inc.to_field::<F>(), odd.rd_inc.to_field::<F>());
            Ok((
                ram_even + challenge * (ram_odd - ram_even),
                rd_even + challenge * (rd_odd - rd_even),
            ))
        };
        // Both tables are written in place, in one row pass, into allocations
        // of their length. A fallible parallel collect is unindexed: every
        // split stages its entries in vectors of its own, which stay live
        // beside the tables until a serial append copies them in.
        let mut ram: Vec<F> = unsafe_allocate_zero_vec(half);
        let mut rd: Vec<F> = unsafe_allocate_zero_vec(half);
        let write = |(y, (ram_y, rd_y)): (usize, (&mut F, &mut F))| {
            (*ram_y, *rd_y) = bound(y)?;
            Ok::<_, SumcheckError<F>>(())
        };
        #[cfg(feature = "parallel")]
        ram.par_iter_mut()
            .zip(rd.par_iter_mut())
            .enumerate()
            .try_for_each(write)?;
        #[cfg(not(feature = "parallel"))]
        ram.iter_mut()
            .zip(rd.iter_mut())
            .enumerate()
            .try_for_each(write)?;
        Ok((Polynomial::new(ram), Polynomial::new(rd)))
    }

    #[inline]
    fn group_evals(&self, y: usize, ram: (F, F), rd: (F, F)) -> [F; 2] {
        let (a_lo, a_hi) = self.ram_weights.pair(y);
        let (b_lo, b_hi) = self.rd_weights.pair(y);
        [
            a_lo * ram.0 + b_lo * rd.0,
            (a_hi + a_hi - a_lo) * (ram.1 + ram.1 - ram.0)
                + (b_hi + b_hi - b_lo) * (rd.1 + rd.1 - rd.0),
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
            self.bind(challenge)?;
        }
        let evals = match &self.incs {
            IncState::Rows(store) => {
                debug_assert_eq!(self.progress.bound(), 0);
                let half = (1usize << self.progress.total()) / 2;
                let access = store.access();
                let group = |y: usize| -> Result<[F; 2], SumcheckError<F>> {
                    let even: IncRow = access.row(2 * y).map_err(|_| row_unavailable())?;
                    let odd: IncRow = access.row(2 * y + 1).map_err(|_| row_unavailable())?;
                    Ok(self.group_evals(
                        y,
                        (even.ram_inc.to_field(), odd.ram_inc.to_field()),
                        (even.rd_inc.to_field(), odd.rd_inc.to_field()),
                    ))
                };
                #[cfg(feature = "parallel")]
                let evals = (0..half)
                    .into_par_iter()
                    .try_fold(
                        || vec![F::zero(); 2],
                        |mut acc, y| {
                            let group = group(y)?;
                            acc[0] += group[0];
                            acc[1] += group[1];
                            Ok::<_, SumcheckError<F>>(acc)
                        },
                    )
                    .try_reduce(|| vec![F::zero(); 2], |a, b| Ok(merge_evals(a, b)))?;
                #[cfg(not(feature = "parallel"))]
                let evals = {
                    let mut acc = vec![F::zero(); 2];
                    for y in 0..half {
                        let group = group(y)?;
                        acc[0] += group[0];
                        acc[1] += group[1];
                    }
                    acc
                };
                evals
            }
            IncState::Dense { ram, rd } => {
                let half = ram.len() / 2;
                par_sum_pair_groups(half, 2, |acc, y| {
                    let group = self.group_evals(y, pair(ram, y), pair(rd, y));
                    acc[0] += group[0];
                    acc[1] += group[1];
                })
            }
        };

        Ok(round_poly_from_skipped_evals(&evals, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: JoltField> SumcheckKernel<F> for IncKernel<F> {
    type Relation = IncClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<IncClaimReductionOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let IncState::Dense { ram, rd } = &self.incs else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "increment tables absent after full binding",
            });
        };
        Ok(IncClaimReductionOutputClaims {
            ram_inc: ram.evals()[0],
            rd_inc: rd.evals()[0],
        })
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_field::{Fr, Ring};
    use jolt_riscv::JoltTraceRow;
    use jolt_verifier::stages::stage6b::inc_claim_reduction::{
        IncClaimReductionChallenges, IncClaimReductionInputClaims,
    };
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::TraceBackend;

    use super::*;
    use crate::optimized::booleanity::testing::{
        alu_row, load_row, no_op_row, store_row, with_trace_backend,
    };
    use crate::optimized::heap_peak::peak_above_start;
    use crate::optimized::parity::{probe_input_claim, run_lockstep, synthetic_point};
    use crate::ReferenceBackend;

    /// The relation over a `2^log_t`-cycle domain, with its inputs.
    struct Case {
        relation: IncClaimReduction<Fr>,
        challenges: IncClaimReductionChallenges<Fr>,
        claims: IncClaimReductionInputClaims<Fr>,
        points: IncClaimReductionInputClaims<Vec<Fr>>,
    }

    impl Case {
        fn new(log_t: usize) -> Self {
            Self {
                relation: IncClaimReduction::new(
                    TraceDimensions::new(log_t),
                    synthetic_point(log_t, 3),
                    synthetic_point(log_t, 5),
                    synthetic_point(log_t, 7),
                    synthetic_point(log_t, 11),
                ),
                challenges: IncClaimReductionChallenges {
                    gamma: Fr::from_u64(29),
                },
                claims: IncClaimReductionInputClaims::default(),
                points: IncClaimReductionInputClaims::default(),
            }
        }

        fn inputs(&self) -> ProverInputs<'_, Fr, IncClaimReduction<Fr>> {
            ProverInputs {
                relation: &self.relation,
                claims: &self.claims,
                points: &self.points,
                challenges: &self.challenges,
            }
        }
    }

    /// `len` rows whose RAM and register increments follow no period: loads,
    /// stores, register writes and no-ops in a pseudo-random order.
    fn rows(len: usize) -> Vec<JoltTraceRow> {
        (0..len as u64)
            .map(
                |cycle| match cycle.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 62 {
                    0 => load_row(0x8000_1000),
                    1 => store_row(0x8000_1008),
                    2 => alu_row(cycle),
                    _ => no_op_row(),
                },
            )
            .collect()
    }

    /// The kernel proves the reference kernel's rounds and openings over
    /// `backend`'s trace.
    fn assert_matches_reference(backend: &TraceBackend) {
        let log_t = backend.config.log_t;
        let case = Case::new(log_t);
        let mut session = ProofSession::default();
        let mut reference =
            <ReferenceBackend as PrepareKernel<Fr, IncClaimReduction<Fr>>>::prepare(
                &ReferenceBackend,
                &mut session,
                backend,
                case.inputs(),
            )
            .unwrap();
        let mut optimized = OptimizedIncClaimReduction
            .prepare(&mut session, backend, case.inputs())
            .unwrap();
        let claim = probe_input_claim(reference.as_mut());
        run_lockstep(
            reference.as_mut(),
            optimized.as_mut(),
            claim,
            &synthetic_point(log_t, 401),
        );
        assert_eq!(
            reference.output_claims(&case.claims).unwrap(),
            optimized.output_claims(&case.claims).unwrap()
        );
    }

    /// On the canned trace, and on a trace that ends inside its domain and
    /// one that fills it, whose first bind fills its tables across parallel
    /// splits.
    #[test]
    fn inc_claim_reduction_matches_reference() {
        with_sample_backend(assert_matches_reference);
        for len in [1500, 2048] {
            with_trace_backend(11, 8, rows(len), |backend, _| {
                assert_matches_reference(backend);
            });
        }
    }

    /// The first bind holds its two `T/2`-entry tables and no vector of a
    /// collect's beside them.
    #[test]
    fn first_bind_holds_only_its_tables() {
        const LOG_T: usize = 12;
        with_trace_backend(LOG_T, 8, rows(1 << LOG_T), |backend, _| {
            let case = Case::new(LOG_T);
            let kernel = || IncKernel::new(backend, case.inputs()).unwrap();
            let challenge = Fr::from_u64(31);
            // Keeps a first bind's one-time allocations out of the measurement.
            kernel().bind(challenge).unwrap();
            let mut kernel = kernel();
            let ((), peak) = peak_above_start(|| kernel.bind(challenge).unwrap());
            let tables = 2 * (1 << (LOG_T - 1)) * size_of::<Fr>();
            assert!(
                (tables..tables + tables / 8).contains(&peak),
                "the first bind held {peak} bytes for {tables} bytes of tables"
            );
        });
    }
}
