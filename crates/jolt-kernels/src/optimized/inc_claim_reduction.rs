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
//! and the summand is `A·RamInc + B·RdInc`: two fused multiplies per point
//! with field-identical round messages. Memory shape (none of the four dense
//! `T`-length field tables the naive port materializes ever exists):
//!
//! - `A` and `B` are served from [`PairedEq`] split tables (~4·√T each): each
//!   eq term factors as `hi[j_hi]·lo[j_lo]`, low-to-high binding touches only
//!   the `j_lo` tensor factor, and the exhausted lo scalar folds into the hi
//!   table (the [`super::support::SplitLt`] argument, per term).
//! - The `RamInc`/`RdInc` columns are read straight from the typed trace
//!   rows for the first round — the same single-sourced extractors behind
//!   `oracle_table`, so the values are identical — and the dense bound
//!   tables (T/2 field elements) appear only at the first bind (the
//!   stage-5 registers val-evaluation deferral, taken one round further).
//!
//! Eval-at-1 recovery and rayon walks as per the [`crate::optimized`]
//! module docs.

use jolt_claims::protocols::jolt::geometry::claim_reductions::increments::{
    ram_inc_reduced, rd_inc_reduced,
};
use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_field::JoltField;
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
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

/// Stage-6b increment claim reduction: `PrepareKernel` front of the
/// optimized kernel.
pub struct OptimizedIncClaimReduction;

/// The two committed increment columns of one cycle.
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
        // A = eq(ram rw) + γ·eq(ram val); B = γ²·eq(reg rw) + γ³·eq(reg val).
        let ram_weights = PairedEq::new(cycle_points[0], F::one(), cycle_points[1], gamma);
        let rd_weights = PairedEq::new(
            cycle_points[2],
            gamma_squared,
            cycle_points[3],
            gamma_squared * gamma,
        );

        let incs = if relation.rounds() == 0 {
            // Single-cycle domain: no bind ever happens, so serve the
            // (one-entry) dense tables from prepare.
            let dense = |id: JoltOpeningId| witness.oracle_table(id.polynomial_id());
            IncState::Dense {
                ram: Polynomial::new(dense(ram_inc_reduced())?),
                rd: Polynomial::new(dense(rd_inc_reduced())?),
            }
        } else {
            IncState::Rows(BundleStore::resolve(witness, cycles)?)
        };

        Ok(Box::new(IncKernel {
            progress: RoundProgress::new(relation.rounds()),
            incs,
            ram_weights,
            rd_weights,
        }))
    }
}

/// `s₁·eq(p₁, ·) + s₂·eq(p₂, ·)` served from four ~√T split tables and bound
/// low-to-high; the scales ride in the hi tables.
///
/// Big-endian index `j = j_hi ‖ j_lo`: each term factors as
/// `hi[j_hi]·lo[j_lo]`, adjacent low-to-high pairs share `j_hi`, and binding
/// acts linearly on the `j_lo` tensor factor — so served values equal the
/// dense combined table bound identically (the [`super::support::SplitLt`]
/// argument, applied per term). Once the lo variables are exhausted the lo
/// scalars fold into one dense hi-sized table.
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F")
)]
enum PairedEq<F> {
    Split {
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
        lo1: Vec<F>,
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
        hi1: Vec<F>,
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
        lo2: Vec<F>,
        #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
        hi2: Vec<F>,
    },
    Dense(#[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))] Vec<F>),
}

impl<F: JoltField> PairedEq<F> {
    fn new(p1: &[F], s1: F, p2: &[F], s2: F) -> Self {
        debug_assert_eq!(p1.len(), p2.len());
        let mid = p1.len() / 2;
        if mid == 0 {
            // 0 or 1 variables: the dense combined table is at most two
            // entries.
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
    fn pair(&self, y: usize) -> (F, F) {
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

    fn bind(&mut self, r: F) {
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
                    // Lo variables exhausted: fold the lo scalars into the
                    // hi tables and continue densely.
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
}

/// The increment columns' lifecycle: typed trace rows until the first bind
/// (the full-length dense field tables never exist), dense bound tables
/// afterwards.
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
enum IncState<F: JoltField> {
    Rows(BundleStore<IncRow>),
    Dense {
        ram: Polynomial<F>,
        rd: Polynomial<F>,
    },
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
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

    /// The first bind over the typed rows: promotes and combines each
    /// vertical pair directly into the bound dense tables — the same values
    /// the dense-table bind produces, without the full-length tables.
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
        #[cfg(feature = "parallel")]
        let (ram, rd) = (0..half)
            .into_par_iter()
            .map(bound)
            .collect::<Result<(Vec<F>, Vec<F>), _>>()?;
        #[cfg(not(feature = "parallel"))]
        let (ram, rd) = {
            let mut ram = Vec::with_capacity(half);
            let mut rd = Vec::with_capacity(half);
            for y in 0..half {
                let (ram_y, rd_y) = bound(y)?;
                ram.push(ram_y);
                rd.push(rd_y);
            }
            (ram, rd)
        };
        Ok((Polynomial::new(ram), Polynomial::new(rd)))
    }

    /// The summand's evaluations at `t ∈ {0, 2}` for group `y`, from the
    /// increment columns' `(lo, hi)` pairs.
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
                            Ok(acc)
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
