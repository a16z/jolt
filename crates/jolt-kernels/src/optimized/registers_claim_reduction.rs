//! The optimized registers claim-reduction (stage 3) kernel.
//!
//! Byte-parity contract: identical round polynomials and output claims to the
//! reference kernel (`reference/registers_claim_reduction.rs`), which binds
//! four dense `T`-sized tables (`eq(τ_low)` plus the three value columns)
//! every round.
//!
//! Carries forward the former registers claim-reduction optimizations:
//!
//! - **Prefix–suffix P·Q decomposition** (eprint 2025/611, Appendix A) for
//!   the first half of the rounds: `Σ_j eq(τ, j)·V(j)` becomes
//!   `Σ_{x_lo} P(x_lo)·Q(x_lo)` with `P = eq(τ_lo)` and
//!   `Q(x_lo) = Σ_{x_hi} eq(τ_hi)[x_hi]·V(x_hi‖x_lo)` — two √T-sized tables
//!   per round instead of four T-sized ones. Since `eq` factors across the
//!   variable split and suffix-summing commutes with prefix partial
//!   evaluation, every P·Q round polynomial equals the dense summand's
//!   exactly.
//! - **Phase-2 regeneration**: once the prefix is bound, the three value
//!   columns are folded by `eq(r_prefix)` straight from the raw `u64` trace
//!   values into `2^(remaining)`-sized tables, and the eq table is the
//!   suffix table scaled by the bound-prefix eq factor — the exact partial
//!   binds of the dense tables, recomputed instead of carried.
//! - **Raw `u64` trace values with small-scalar deferred-reduction
//!   accumulation** (`fmadd_u64`) for the Q build and the phase-2 folds.
//! - **Two task-partitioned row passes** (`task_ranges`): the Q build and the
//!   phase-2 fold each read the cycle domain once, every task visiting a
//!   contiguous run of whole `x_hi` rows serially. The Q build accumulates
//!   into one caller-allocated `2^prefix`-entry slot per task, merged in place
//!   before reducing; the phase-2 fold writes each completed `x_hi` row into
//!   its task's slices of the tables.

use std::ops::Range;

use jolt_claims::protocols::jolt::{
    JoltDerivedId, JoltPolynomialId, RegistersClaimReductionPublic,
};
use jolt_field::{Accumulator, JoltField};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage3::registers_claim_reduction::{
    RegistersClaimReduction, RegistersClaimReductionOutputClaims,
};
use jolt_witness::__private::TraceRow;
use jolt_witness::witnesses::WitnessEnv;
use jolt_witness::{JoltWitnessPlane, TraceRows, WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::support::{
    bind_pairs, fmadd_u64_split, map_indices, pin_derived_term, split_at_tasks, task_ranges,
    RoundChallenges, VISIT_ROWS,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Per-cycle `[rd write value, rs1 value, rs2 value]`, kept as raw `u64`s so
/// the eq folds run on small-scalar fused multiply-adds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct RegisterValuesRow([u64; 3]);

impl WitnessBundle for RegisterValuesRow {
    fn from_row(
        row: &TraceRow,
        _next: Option<&TraceRow>,
        _env: &WitnessEnv<'_>,
    ) -> Result<Self, WitnessError> {
        Ok(Self([
            row.rd_write_value(),
            row.rs1_value(),
            row.rs2_value(),
        ]))
    }

    fn annotated_ids() -> Vec<JoltPolynomialId> {
        Vec::new()
    }
}

pub struct OptimizedRegistersClaimReduction;

impl<F: JoltField> PrepareKernel<F, RegistersClaimReduction<F>>
    for OptimizedRegistersClaimReduction
{
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RegistersClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersClaimReduction<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let log_t = relation.rounds();
        if log_t == 0 {
            return Err(KernelError::Unsupported {
                reason: "optimized registers claim reduction requires at least one cycle round",
            });
        }
        let tau: &[F] = relation.product_uniskip_tau_low();
        if tau.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "registers claim-reduction tau point has the wrong variable count",
            });
        }
        const LABEL: &str = "registers claim-reduction trace rows";
        let cycles = 1usize << log_t;
        let rows = witness
            .rows()
            .ok_or(WitnessError::UnavailableView { label: LABEL })?;
        if rows.cycles() != cycles {
            return Err(WitnessError::InvalidDimensions {
                label: LABEL,
                reason: format!("{} rows for a 2^{log_t}-cycle stage", rows.cycles()),
            }
            .into());
        }

        let gamma = inputs.challenges.gamma;
        let gamma_sq = gamma * gamma;

        let (tau_hi, tau_lo) = tau.split_at(log_t / 2);
        let prefix_vars = tau_lo.len();
        let p = EqPolynomial::<F>::evals(tau_lo, None);
        let eq_suffix: Vec<[F; 2]> = EqPolynomial::<F>::evals(tau_hi, None)
            .into_iter()
            .map(|eq_hi| [eq_hi, eq_hi.mul_pow_2(32)])
            .collect();

        // Each task visits a contiguous run of whole `x_hi` rows and adds every
        // row into entry `x_lo` of its own slot, allocated before the pass, so
        // partial state is `tasks · 2^prefix_vars` entries whatever the
        // scheduling. The merge gives each accumulator exactly the split terms
        // one accumulator per `x_lo` would take: its headroom is unchanged.
        let x_lo_mask = (1 << prefix_vars) - 1;
        let ranges = task_ranges(cycles, 1 << prefix_vars);
        let mut slots = map_indices(ranges.len(), |_| {
            vec![[F::SmallScalarAccumulator::default(); 3]; 1 << prefix_vars]
        });
        let accumulate =
            |(slot, range): (&mut Vec<[F::SmallScalarAccumulator; 3]>, &Range<usize>)| {
                rows.visit_bundles(
                    range.clone(),
                    VISIT_ROWS,
                    |first, bundles: &[RegisterValuesRow]| {
                        for (cycle, row) in (first..).zip(bundles) {
                            let [eq_hi, eq_hi_shifted] = eq_suffix[cycle >> prefix_vars];
                            let fold = &mut slot[cycle & x_lo_mask];
                            fmadd_u64_split(&mut fold[0], eq_hi, eq_hi_shifted, row.0[0]);
                            fmadd_u64_split(&mut fold[1], eq_hi, eq_hi_shifted, row.0[1]);
                            fmadd_u64_split(&mut fold[2], eq_hi, eq_hi_shifted, row.0[2]);
                        }
                        Ok(())
                    },
                )
            };
        #[cfg(feature = "parallel")]
        slots.par_iter_mut().zip(&ranges).try_for_each(accumulate)?;
        #[cfg(not(feature = "parallel"))]
        slots.iter_mut().zip(&ranges).try_for_each(accumulate)?;

        let (merged, rest) = slots.split_at_mut(1);
        let merged = &mut merged[0];
        let rest = &*rest;
        let merge = |(x_lo, fold): (usize, &mut [F::SmallScalarAccumulator; 3])| {
            for slot in rest {
                for (sum, term) in fold.iter_mut().zip(slot[x_lo]) {
                    sum.merge(term);
                }
            }
        };
        #[cfg(feature = "parallel")]
        merged.par_iter_mut().enumerate().for_each(merge);
        #[cfg(not(feature = "parallel"))]
        merged.iter_mut().enumerate().for_each(merge);
        let q: Vec<F> = merged
            .iter()
            .map(|fold| fold[0].reduce() + gamma * fold[1].reduce() + gamma_sq * fold[2].reduce())
            .collect();

        Ok(Box::new(ClaimReductionKernel {
            log_t,
            gamma,
            gamma_sq,
            tau: tau.to_vec(),
            phase: Phase::PrefixSuffix { p, q, rows },
            challenges: RoundChallenges::new(log_t),
        }))
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum Phase<F> {
    /// First half of the rounds: the P·Q buffers over the prefix variables,
    /// and the rows the transition folds.
    PrefixSuffix {
        p: Vec<F>,
        q: Vec<F>,
        /// The witness plane owns these rows; it reports them itself.
        #[cfg_attr(feature = "allocative", allocative(skip))]
        rows: TraceRows,
    },
    /// Remaining rounds: the eq table and the three value columns, dense.
    Dense {
        eq: Vec<F>,
        rd_write_value: Vec<F>,
        rs1_value: Vec<F>,
        rs2_value: Vec<F>,
    },
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct ClaimReductionKernel<F: JoltField> {
    log_t: usize,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gamma: F,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    gamma_sq: F,
    tau: Vec<F>,
    phase: Phase<F>,
    challenges: RoundChallenges<F>,
}

impl<F: JoltField> ClaimReductionKernel<F> {
    /// The dense phase at the bound prefix: one pass in which every task folds
    /// whole `x_hi` rows by `eq(r_prefix, x_lo)` and writes each row's three
    /// values into its own slices of the tables.
    fn dense_phase(&self, rows: &TraceRows) -> Result<Phase<F>, WitnessError> {
        let bound = self.challenges.bound();
        let r_prefix: Vec<F> = self.challenges.as_slice().iter().rev().copied().collect();
        let eq_prefix = EqPolynomial::<F>::evals(&r_prefix, None);
        let eq_prefix_shifted: Vec<F> = eq_prefix.iter().map(|eq| eq.mul_pow_2(32)).collect();
        let cycles = 1usize << self.log_t;
        let x_lo_mask = (1 << bound) - 1;

        let mut columns: [Vec<F>; 3] = core::array::from_fn(|_| vec![F::zero(); cycles >> bound]);
        let ranges = task_ranges(cycles, 1 << bound);
        let slices = split_at_tasks(
            columns.each_mut().map(Vec::as_mut_slice),
            &ranges,
            |cycle| cycle >> bound,
        );
        let fold = |(range, mut outputs): (&Range<usize>, [&mut [F]; 3])| {
            let first_row = range.start >> bound;
            let mut folds = [F::SmallScalarAccumulator::default(); 3];
            rows.visit_bundles(
                range.clone(),
                VISIT_ROWS,
                |first, bundles: &[RegisterValuesRow]| {
                    for (cycle, row) in (first..).zip(bundles) {
                        let x_lo = cycle & x_lo_mask;
                        let (eq, eq_shifted) = (eq_prefix[x_lo], eq_prefix_shifted[x_lo]);
                        fmadd_u64_split(&mut folds[0], eq, eq_shifted, row.0[0]);
                        fmadd_u64_split(&mut folds[1], eq, eq_shifted, row.0[1]);
                        fmadd_u64_split(&mut folds[2], eq, eq_shifted, row.0[2]);
                        if x_lo == x_lo_mask {
                            let entry = (cycle >> bound) - first_row;
                            for (output, fold) in outputs.iter_mut().zip(std::mem::take(&mut folds))
                            {
                                output[entry] = fold.reduce();
                            }
                        }
                    }
                    Ok(())
                },
            )
        };
        #[cfg(feature = "parallel")]
        ranges.par_iter().zip(slices).try_for_each(fold)?;
        #[cfg(not(feature = "parallel"))]
        ranges.iter().zip(slices).try_for_each(fold)?;

        let (tau_hi, tau_lo) = self.tau.split_at(self.log_t / 2);
        let eq_prefix_eval = EqPolynomial::<F>::mle(&r_prefix, tau_lo);
        let [rd_write_value, rs1_value, rs2_value] = columns;
        Ok(Phase::Dense {
            eq: EqPolynomial::<F>::evals(tau_hi, Some(eq_prefix_eval)),
            rd_write_value,
            rs1_value,
            rs2_value,
        })
    }

    fn bind(&mut self, r: F) -> Result<(), SumcheckError<F>> {
        self.challenges.push(r);
        if let Phase::PrefixSuffix { p, rows, .. } = &self.phase {
            if p.len() == 2 {
                self.phase =
                    self.dense_phase(rows)
                        .map_err(|_| SumcheckError::MissingEvaluationSource {
                            kind: "register values row",
                        })?;
                return Ok(());
            }
        }
        match &mut self.phase {
            Phase::PrefixSuffix { p, q, .. } => {
                for table in [p, q] {
                    bind_pairs(table, r);
                }
            }
            Phase::Dense {
                eq,
                rd_write_value,
                rs1_value,
                rs2_value,
            } => {
                for table in [eq, rd_write_value, rs1_value, rs2_value] {
                    bind_pairs(table, r);
                }
            }
        }
        Ok(())
    }
}

impl<F: JoltField> ProveRounds<F> for ClaimReductionKernel<F> {
    fn num_rounds(&self) -> usize {
        self.log_t
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

        // Degree-2 member: evals at t = 0 and t = 2; s(1) from the hint.
        let evals: [F; 2] = match &self.phase {
            Phase::PrefixSuffix { p, q, .. } => {
                let mut acc = [F::Accumulator::default(); 2];
                for y in 0..p.len() / 2 {
                    let (p_0, p_1) = (p[2 * y], p[2 * y + 1]);
                    let (q_0, q_1) = (q[2 * y], q[2 * y + 1]);
                    acc[0].fmadd(p_0, q_0);
                    acc[1].fmadd(p_1 + p_1 - p_0, q_1 + q_1 - q_0);
                }
                acc.map(F::Accumulator::reduce)
            }
            Phase::Dense {
                eq,
                rd_write_value,
                rs1_value,
                rs2_value,
            } => {
                let mut acc = [F::Accumulator::default(); 2];
                for y in 0..eq.len() / 2 {
                    let pair = |table: &[F]| (table[2 * y], table[2 * y + 1]);
                    let (eq_0, eq_1) = pair(eq);
                    let (rd_0, rd_1) = pair(rd_write_value);
                    let (rs1_0, rs1_1) = pair(rs1_value);
                    let (rs2_0, rs2_1) = pair(rs2_value);
                    acc[0].fmadd(eq_0, rd_0 + self.gamma * rs1_0 + self.gamma_sq * rs2_0);
                    acc[1].fmadd(
                        eq_1 + eq_1 - eq_0,
                        (rd_1 + rd_1 - rd_0)
                            + self.gamma * (rs1_1 + rs1_1 - rs1_0)
                            + self.gamma_sq * (rs2_1 + rs2_1 - rs2_0),
                    );
                }
                acc.map(F::Accumulator::reduce)
            }
        };

        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: JoltField> SumcheckKernel<F> for ClaimReductionKernel<F> {
    type Relation = RegistersClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<RegistersClaimReductionOutputClaims<F>, SumcheckKernelError<F>> {
        self.challenges.require_complete()?;
        let Phase::Dense {
            rd_write_value,
            rs1_value,
            rs2_value,
            ..
        } = &self.phase
        else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "claim reduction must finish in the dense phase",
            });
        };
        Ok(RegistersClaimReductionOutputClaims {
            rd_write_value: rd_write_value[0],
            rs1_value: rs1_value[0],
            rs2_value: rs2_value[0],
        })
    }

    /// Pin the regenerated eq table to the verifier's scalar path: its fully
    /// bound value must equal `derive_output_term(EqSpartan)`.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.challenges.require_complete()?;
        let Phase::Dense { eq, .. } = &self.phase else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "claim reduction must finish in the dense phase",
            });
        };
        let id = JoltDerivedId::from(RegistersClaimReductionPublic::EqSpartan);
        pin_derived_term(relation, id, input_points, output_points, challenges, eq[0])
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::{JoltPolynomialId, JoltVirtualPolynomial};
    #[cfg(feature = "akita")]
    use jolt_field::Prime128OffsetA7F7;
    use jolt_field::{Fr, JoltField, Ring};
    use jolt_poly::Polynomial;
    use jolt_verifier::stages::stage3::registers_claim_reduction::{
        RegistersClaimReduction, RegistersClaimReductionChallenges,
        RegistersClaimReductionInputClaims,
    };
    use jolt_witness::{JoltVmWitnessInputs, JoltWitnessOracle, TraceBackend, WitnessError};

    use super::super::registers_read_write::test_support::{
        assert_kernel_parity, assert_nontrivial, challenge_sequence, structured_fixture,
        TraceFixture,
    };
    use super::OptimizedRegistersClaimReduction;
    use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

    /// The optimized kernel against the reference tier over `backend`'s rows.
    fn assert_parity<F: JoltField>(backend: &TraceBackend, log_t: usize, seed: u64) {
        let tau = challenge_sequence::<F>(log_t, seed ^ 0x7E7E);
        let relation = RegistersClaimReduction::<F>::new(TraceDimensions::new(log_t), tau.clone());
        let evaluate = |polynomial: JoltVirtualPolynomial| {
            let table = JoltWitnessOracle::<F>::oracle_table(
                backend,
                JoltPolynomialId::Virtual(polynomial),
            )
            .unwrap();
            Polynomial::new(table).evaluate(&tau)
        };
        let gamma = F::from_u64(0x0DDB_A11C_0FFE_E123);
        let claims = RegistersClaimReductionInputClaims {
            rd_write_value: evaluate(JoltVirtualPolynomial::RdWriteValue),
            rs1_value: evaluate(JoltVirtualPolynomial::Rs1Value),
            rs2_value: evaluate(JoltVirtualPolynomial::Rs2Value),
        };
        let input_claim =
            claims.rd_write_value + gamma * claims.rs1_value + gamma * gamma * claims.rs2_value;
        assert_nontrivial(input_claim);
        let round_challenges = challenge_sequence(log_t, seed);
        assert_kernel_parity(
            &OptimizedRegistersClaimReduction,
            backend,
            &relation,
            &claims,
            &RegistersClaimReductionInputClaims::default(),
            &RegistersClaimReductionChallenges { gamma },
            input_claim,
            &round_challenges,
        );
    }

    fn run_parity(fixture: TraceFixture, log_t: usize, seed: u64) {
        fixture.with_plane(log_t, |backend| assert_parity::<Fr>(backend, log_t, seed));
    }

    /// The optimized kernel prepared over `witness` for an arbitrary relation.
    fn prepare_optimized(
        witness: &TraceBackend,
        log_t: usize,
    ) -> Result<Box<dyn SumcheckKernel<Fr, Relation = RegistersClaimReduction<Fr>>>, KernelError<Fr>>
    {
        let relation = RegistersClaimReduction::<Fr>::new(
            TraceDimensions::new(log_t),
            challenge_sequence(log_t, 5),
        );
        OptimizedRegistersClaimReduction.prepare(
            &mut ProofSession::default(),
            witness,
            ProverInputs {
                relation: &relation,
                claims: &RegistersClaimReductionInputClaims::default(),
                points: &RegistersClaimReductionInputClaims::default(),
                challenges: &RegistersClaimReductionChallenges {
                    gamma: Fr::from_u64(3),
                },
            },
        )
    }

    #[test]
    fn parity_structured_odd_log_t() {
        run_parity(structured_fixture(8), 3, 101);
    }

    #[test]
    fn parity_structured_even_log_t() {
        run_parity(structured_fixture(16), 4, 103);
    }

    #[test]
    fn parity_minimal_single_round() {
        // log_t = 1: the P·Q phase covers the single round and the dense
        // phase materializes inside `finish_rounds`.
        let mut fixture = TraceFixture::new();
        fixture.op(Some(6), Some(2), Some(3));
        fixture.op(None, Some(6), Some(6));
        run_parity(fixture, 1, 107);
    }

    #[test]
    fn parity_padded_trace() {
        let mut fixture = TraceFixture::new();
        fixture.op(Some(8), Some(1), None);
        fixture.op(Some(9), Some(8), Some(8));
        fixture.op(None, Some(9), Some(8));
        run_parity(fixture, 2, 109);
    }

    /// Reference parity over both row passes' task partitions at three
    /// depths; Akita builds also check the Akita field.
    #[test]
    fn rows_match_reference_across_task_passes() {
        for log_t in [1, 3, 6] {
            structured_fixture((1 << log_t) - 1).with_plane(log_t, |resident| {
                assert_parity::<Fr>(resident, log_t, 113);
                #[cfg(feature = "akita")]
                assert_parity::<Prime128OffsetA7F7>(resident, log_t, 127);
            });
        }
    }

    /// A witness without a valid handle or with one of another domain fails
    /// typed.
    #[test]
    fn missing_or_mismatched_rows_are_typed() {
        structured_fixture(7).with_plane(3, |resident| {
            let with_log_t = |log_t| {
                let mut witness = TraceBackend::new(
                    resident.config.clone(),
                    JoltVmWitnessInputs::new(
                        &resident.program,
                        &resident.preprocessing,
                        resident.trace.clone(),
                    ),
                );
                witness.config.log_t = log_t;
                witness
            };
            assert!(matches!(
                prepare_optimized(&with_log_t(1), 3),
                Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
            ));
            assert!(matches!(
                prepare_optimized(&with_log_t(4), 3),
                Err(KernelError::Witness(WitnessError::InvalidDimensions { .. }))
            ));
        });
    }
}
